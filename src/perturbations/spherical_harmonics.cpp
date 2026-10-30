/**
 * Central-body spherical-harmonics gravity
 *
 * Fully normalized Cunningham / Montenbruck-Gill V, W recursion (Montenbruck & Gill sec. 3.2.4),
 * with the acceleration and the gradient from Cunningham's derivative identities.
 * Purely Cartesian, so there is no singularity at the poles. See docs/spherical_harmonics.md.
 */

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cstdint>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/stat.h>
#endif
#include <SpiceUsr.h>          // access SPICE files
#include "perturbations/spherical_harmonics.h"

namespace {

// ----------------------------------------------------------------------------- parsing helpers
std::string trim(const std::string& s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) {
        b++;
    }
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) {
        e--;
    }
    return s.substr(b, e - b);
}

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        tokens.push_back(trim(tok));
    }
    return tokens;
}

std::vector<std::string> split_ws(const std::string& line) {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string tok;
    while (ss >> tok) {
        tokens.push_back(tok);
    }
    return tokens;
}

// strtod after Fortran D -> E exponent replacement (EGM2008 has "1.0d0"); false if not a number
bool to_double(std::string tok, double& value) {
    tok = trim(tok);
    for (char& c : tok) {
        if (c == 'D' || c == 'd') {
            c = 'E';
        }
    }
    if (tok.empty()) {
        return false;
    }
    char* end = nullptr;
    value = std::strtod(tok.c_str(), &end);
    return end == tok.c_str() + tok.size();
}

double parse_double(const std::string& tok, const std::string& path, const std::string& what) {
    double value;
    if (!to_double(tok, value)) {
        throw std::runtime_error(path + ": cannot parse " + what + " from '" + tok + "'");
    }
    return value;
}

int parse_int(const std::string& tok, const std::string& path, const std::string& what) {
    const std::string t = trim(tok);
    char* end = nullptr;
    long value = std::strtol(t.c_str(), &end, 10);
    if (t.empty() || end != t.c_str() + t.size()) {
        throw std::runtime_error(path + ": cannot parse " + what + " from '" + tok + "'");
    }
    return static_cast<int>(value);
}

void strip_cr(std::string& line) {
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
}

// Remove a UTF-8 byte order mark at the start of the first line
void strip_bom(std::string& line) {
    if (line.compare(0, 3, "\xEF\xBB\xBF") == 0) {
        line.erase(0, 3);
    }
}

// ----------------------------------------------------------------------------- file formats
enum class GravityFormat { shadr, icgem };

// True if line has 8 comma-separated numbers (PDS SHADR header)
bool is_shadr_header(const std::string& line) {
    std::vector<std::string> fields = split_csv(line);
    if (fields.size() != 8) {
        return false;
    }
    double value;
    for (const std::string& f : fields) {
        if (!to_double(f, value)) {
            return false;
        }
    }
    return true;
}

/**
 * PDS SHADR if the first line has 8 comma-separated numbers, otherwise ICGEM
 * (load_icgem throws if the file has no end_of_head line).
 */
GravityFormat detect_format(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open gravity field file " + path);
    }
    std::string first;
    std::getline(in, first);
    strip_cr(first);
    strip_bom(first);
    return is_shadr_header(first) ? GravityFormat::shadr : GravityFormat::icgem;
}

// Store one coefficient row; rows with n > nmax are ignored by the callers
void store_row(SphericalHarmonicsModel& model, std::vector<char>& seen,
               int n, int m, double C, double S, const std::string& path) {
    if (n < 0 || m < 0 || m > n) {
        throw std::runtime_error(path + ": invalid degree/order (" + std::to_string(n) + ", "
                                 + std::to_string(m) + ")");
    }
    if (!std::isfinite(C) || !std::isfinite(S)) {
        throw std::runtime_error(path + ": non-finite coefficient at (" + std::to_string(n) + ", "
                                 + std::to_string(m) + ")");
    }
    const int k = sh_index(n, m);
    model.C[k] = C;
    model.S[k] = S;
    seen[k] = 1;
}

/**
 * Check nmax against the degree of the file and resolve the truncation order:
 * mmax < 0 means min(nmax, order of the file); mmax above the order of the file throws.
 */
void resolve_truncation(SphericalHarmonicsModel& model, int mmax, const std::string& path) {
    if (model.file_order < 0 || model.file_order > model.file_degree) {
        throw std::runtime_error(path + ": invalid order of the field, " + std::to_string(model.file_order));
    }
    if (model.nmax > model.file_degree) {
        throw std::invalid_argument(path + ": nmax = " + std::to_string(model.nmax)
                                    + " exceeds the degree of the field, " + std::to_string(model.file_degree));
    }
    if (mmax < 0) {
        model.mmax = std::min(model.nmax, model.file_order);
    } else if (mmax > model.file_order) {
        throw std::invalid_argument(path + ": mmax = " + std::to_string(mmax)
                                    + " exceeds the order of the field, " + std::to_string(model.file_order));
    } else {
        model.mmax = mmax;
    }
}

// Number of (n, m) with 2 <= n <= nmax and m <= min(n, mmax)
int count_needed(const SphericalHarmonicsModel& model) {
    int count = 0;
    for (int n = 2; n <= model.nmax; n++) {
        count += std::min(n, model.mmax) + 1;
    }
    return count;
}

/**
 * PDS SHADR (GRAIL .tab): header line "R [km], GM [km^3/s^2], sigma GM, degree, order,
 * normalization state, ref. lon, ref. lat", then rows "n, m, C, S, sigma C, sigma S"
 * sorted by n, starting at n = 1.
 */
void load_shadr(const std::string& path, SphericalHarmonicsModel& model, std::vector<char>& seen, int mmax) {
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);
    strip_cr(line);
    strip_bom(line);
    std::vector<std::string> hdr = split_csv(line);
    // header errors say how the format was chosen (a .gfc whose first line is 8 numbers is read as .tab)
    const std::string note = " (file detected as PDS SHADR .tab from its first line)";
    try {
        model.R_km = parse_double(hdr[0], path, "reference radius");
        model.GM_km3s2 = parse_double(hdr[1], path, "GM");
        model.file_degree = parse_int(hdr[3], path, "degree of field");
        model.file_order = parse_int(hdr[4], path, "order of field");
        const int norm = parse_int(hdr[5], path, "normalization state");
        if (norm != 1) {
            throw std::runtime_error(path + ": normalization state " + std::to_string(norm)
                                     + " is not supported; only fully normalized (1) coefficients are");
        }
        resolve_truncation(model, mmax, path);
    } catch (const std::invalid_argument& ex) {
        throw std::invalid_argument(ex.what() + note);
    } catch (const std::runtime_error& ex) {
        throw std::runtime_error(ex.what() + note);
    }
    bool warned_order = false;
    while (std::getline(in, line)) {
        strip_cr(line);
        if (trim(line).empty()) {
            continue;
        }
        std::vector<std::string> tok = split_csv(line);
        if (tok.size() < 4) {
            throw std::runtime_error(path + ": malformed coefficient line '" + line + "'");
        }
        const int n = parse_int(tok[0], path, "degree");
        const int m = parse_int(tok[1], path, "order");
        if (n > model.nmax) {
            break;   // sorted by degree: do not read the rest of a large file
        }
        if (m > model.file_order && !warned_order) {
            std::cerr << "Warning: " << path << " has rows with order above the order of the field in its header ("
                      << model.file_order << "); orders above " << model.mmax << " are not used." << std::endl;
            warned_order = true;
        }
        store_row(model, seen, n, m, parse_double(tok[2], path, "C"), parse_double(tok[3], path, "S"), path);
    }
}

/**
 * ICGEM (.gfc): optional free text, optional begin_of_head, "keyword value" header lines until
 * end_of_head, then rows "gfc n m C S sigma_C sigma_S". GM in m^3/s^2 and radius in m.
 * Leading blanks are ignored. Header lines that are not "keyword value" (e.g. comments
 * starting with a keyword) are skipped: numeric keywords need a numeric value, the other
 * keywords exactly one value.
 */
void load_icgem(const std::string& path, SphericalHarmonicsModel& model, std::vector<char>& seen, int mmax) {
    std::ifstream in(path);
    std::string line;
    std::map<std::string, double> hdr_num;        // earth_gravity_constant, radius, max_degree
    std::map<std::string, std::string> hdr_str;   // norm, tide_system
    bool end_of_head = false;
    bool first_line = true;
    while (std::getline(in, line)) {
        strip_cr(line);
        if (first_line) {
            strip_bom(line);
            first_line = false;
        }
        std::vector<std::string> tok = split_ws(line);
        if (tok.empty()) {
            continue;
        }
        if (tok[0] == "end_of_head") {
            end_of_head = true;
            break;
        }
        if (tok[0] == "begin_of_head") {
            hdr_num.clear();      // everything before begin_of_head is free text
            hdr_str.clear();
            continue;
        }
        double value;
        const bool numeric_key = (tok[0] == "earth_gravity_constant" || tok[0] == "radius" || tok[0] == "max_degree")
                                 && tok.size() >= 2 && to_double(tok[1], value);
        const bool string_key = (tok[0] == "norm" || tok[0] == "tide_system") && tok.size() == 2;
        if ((numeric_key || string_key) && (hdr_num.count(tok[0]) || hdr_str.count(tok[0]))) {
            throw std::runtime_error(path + ": header keyword " + tok[0] + " appears more than once");
        }
        if (numeric_key) {
            hdr_num[tok[0]] = value;
        } else if (string_key) {
            hdr_str[tok[0]] = tok[1];
        }
    }
    if (!end_of_head) {
        throw std::runtime_error(path + ": not a PDS SHADR (.tab) or ICGEM (.gfc) gravity field file"
                                 " (the first line is not an 8-field SHADR header, and there is no"
                                 " end_of_head line)");
    }
    for (const char* key : {"earth_gravity_constant", "radius", "max_degree"}) {
        if (hdr_num.find(key) == hdr_num.end()) {
            throw std::runtime_error(path + ": header keyword " + key + " missing");
        }
    }
    model.GM_km3s2 = hdr_num["earth_gravity_constant"] / 1e9;
    model.R_km = hdr_num["radius"] / 1e3;
    const double max_degree = hdr_num["max_degree"];
    if (!(max_degree >= 0 && max_degree <= 1e6 && max_degree == std::floor(max_degree))) {
        throw std::runtime_error(path + ": invalid max_degree");
    }
    model.file_degree = static_cast<int>(max_degree);
    model.file_order = model.file_degree;   // ICGEM has no order keyword; missing rows are reported below
    // ICGEM default normalization is fully_normalized (EGM96.gfc has no norm line)
    if (hdr_str.count("norm") && hdr_str["norm"] != "fully_normalized") {
        throw std::runtime_error(path + ": norm " + hdr_str["norm"]
                                 + " is not supported; only fully_normalized coefficients are");
    }
    if (hdr_str.count("tide_system")) {
        model.tide_system = hdr_str["tide_system"];
    }
    resolve_truncation(model, mmax, path);

    // ICGEM does not guarantee the order of the rows: read until every needed row was seen
    const int n_needed = count_needed(model);
    int n_seen = 0;
    while (std::getline(in, line)) {
        strip_cr(line);
        std::vector<std::string> tok = split_ws(line);
        if (tok.empty()) {
            continue;
        }
        if (tok[0] == "gfct" || tok[0] == "trnd" || tok[0] == "acos" || tok[0] == "asin") {
            throw std::runtime_error(path + ": time-variable ICGEM models (" + tok[0]
                                     + " rows) are not supported");
        }
        if (tok[0] != "gfc") {
            continue;
        }
        if (tok.size() < 5) {
            throw std::runtime_error(path + ": malformed coefficient line '" + line + "'");
        }
        const int n = parse_int(tok[1], path, "degree");
        const int m = parse_int(tok[2], path, "order");
        if (n > model.nmax) {
            if (n_seen == n_needed) {
                break;   // everything needed was read; the rest are higher degrees
            }
            continue;
        }
        if (n >= 2 && m >= 0 && m <= std::min(n, model.mmax) && !seen[sh_index(n, m)]) {
            n_seen++;
        }
        store_row(model, seen, n, m, parse_double(tok[3], path, "C"), parse_double(tok[4], path, "S"), path);
    }
}

/**
 * Recursion factors (n <= nmax + 2) and derivative factors (n <= nmax), docs/spherical_harmonics.md
 */
void precompute_factors(SphericalHarmonicsModel& model) {
    const int nmax = model.nmax;
    const int N = nmax + 2;

    // V, W recursion: sectoral c_m, first off-diagonal a_{m+1,m} = sqrt(2m+3), column a_nm, b_nm
    model.rec_a.assign(sh_index(N + 1, 0), 0.0);
    model.rec_b.assign(sh_index(N + 1, 0), 0.0);
    model.rec_c.assign(N + 1, 0.0);
    for (int m = 1; m <= N; m++) {
        model.rec_c[m] = (m == 1) ? std::sqrt(3.0) : std::sqrt((2.0*m + 1) / (2.0*m));
    }
    for (int m = 0; m <= N; m++) {
        if (m + 1 <= N) {
            model.rec_a[sh_index(m + 1, m)] = std::sqrt(2.0*m + 3);
        }
        for (int n = m + 2; n <= N; n++) {
            const double nn = n, mm = m;
            model.rec_a[sh_index(n, m)] = std::sqrt((2*nn + 1)*(2*nn - 1) / ((nn - mm)*(nn + mm)));
            model.rec_b[sh_index(n, m)] = std::sqrt((2*nn + 1)*(nn + mm - 1)*(nn - mm - 1)
                                                    / ((2*nn - 3)*(nn + mm)*(nn - mm)));
        }
    }

    // acceleration (FA*) and gradient (FG*) factors; d_m = 1 if m = 0 else 2
    const int size = sh_index(nmax + 1, 0);
    for (std::vector<double>* f : {&model.FAz, &model.FAp, &model.FAm, &model.FG0,
                                   &model.FGpp, &model.FGp, &model.FGm, &model.FGmm}) {
        f->assign(size, 0.0);
    }
    for (int n = 0; n <= nmax; n++) {
        for (int m = 0; m <= n; m++) {
            const int k = sh_index(n, m);
            const double nn = n, mm = m;
            const double dm = (m == 0) ? 1.0 : 2.0;
            const double a3 = (2*nn + 1) / (2*nn + 3);
            const double a5 = (2*nn + 1) / (2*nn + 5);
            model.FAz[k] = std::sqrt(a3*(nn - mm + 1)*(nn + mm + 1));
            model.FAp[k] = std::sqrt(dm/2*a3*(nn + mm + 1)*(nn + mm + 2));
            model.FG0[k] = std::sqrt(a5*(nn - mm + 1)*(nn - mm + 2)*(nn + mm + 1)*(nn + mm + 2));
            model.FGpp[k] = std::sqrt(dm/2*a5*(nn + mm + 1)*(nn + mm + 2)*(nn + mm + 3)*(nn + mm + 4));
            model.FGp[k] = std::sqrt(dm/2*a5*(nn - mm + 1)*(nn + mm + 1)*(nn + mm + 2)*(nn + mm + 3));
            if (m >= 1) {
                const double ratio1 = (m == 1) ? 2.0 : 1.0;                 // d_m / d_{m-1}
                const double ratio2 = (m == 2) ? 2.0 : 1.0;                 // rho_m: 1, 2, 1, 1, ...
                model.FAm[k] = std::sqrt(ratio1*a3*(nn - mm + 1)*(nn - mm + 2));
                model.FGm[k] = std::sqrt(ratio1*a5*(nn - mm + 1)*(nn - mm + 2)*(nn - mm + 3)*(nn + mm + 1));
                model.FGmm[k] = std::sqrt(ratio2*a5*(nn - mm + 1)*(nn - mm + 2)*(nn - mm + 3)*(nn - mm + 4));
            }
        }
    }
}

}  // namespace


SphericalHarmonicsWorkspace::SphericalHarmonicsWorkspace(int nmax)
    : V(sh_index(nmax + 3, 0), 0.0), W(sh_index(nmax + 3, 0), 0.0) {}


/**
 * Load a gravity field file, truncated to degree nmax and order mmax
 *
 * @param path PDS SHADR .tab (GRAIL) or ICGEM .gfc (EGM2008, EGM96) file; the format is auto-detected
 * @param nmax truncation degree, 2 <= nmax <= min(file degree, SH_NMAX_LIMIT)
 * @param mmax truncation order, mmax <= min(nmax, order of the file); negative means min(nmax, order of the file)
 * @return model with GM in km^3/s^2, R in km, and precomputed factors
 */
SphericalHarmonicsModel load_spherical_harmonics(const std::string& path, int nmax, int mmax) {
    if (nmax < 2) {
        throw std::invalid_argument("spherical harmonics: nmax must be >= 2, got " + std::to_string(nmax));
    }
    if (nmax > SH_NMAX_LIMIT) {
        throw std::invalid_argument("spherical harmonics: nmax > " + std::to_string(SH_NMAX_LIMIT)
                                    + " is not supported");
    }
    if (mmax > nmax) {
        throw std::invalid_argument("spherical harmonics: mmax must be <= nmax");
    }

    SphericalHarmonicsModel model;
    model.source = path;
    model.nmax = nmax;
    model.C.assign(sh_index(nmax + 1, 0), 0.0);
    model.S.assign(sh_index(nmax + 1, 0), 0.0);
    std::vector<char> seen(sh_index(nmax + 1, 0), 0);

    if (detect_format(path) == GravityFormat::icgem) {
        load_icgem(path, model, seen, mmax);
    } else {
        load_shadr(path, model, seen, mmax);
    }
    mmax = model.mmax;    // resolved against the order of the file
    if (!(model.GM_km3s2 > 0) || !(model.R_km > 0) || !std::isfinite(model.GM_km3s2) || !std::isfinite(model.R_km)) {
        throw std::runtime_error(path + ": GM and reference radius must be positive");
    }

    // every (n, m) with 2 <= n <= nmax and m <= mmax must be present
    for (int n = 2; n <= nmax; n++) {
        for (int m = 0; m <= std::min(n, mmax); m++) {
            if (!seen[sh_index(n, m)]) {
                throw std::runtime_error(path + ": coefficient (" + std::to_string(n) + ", "
                                         + std::to_string(m) + ") missing (truncated file, or an "
                                         "order-limited field: pass a smaller mmax)");
            }
        }
    }
    // degree 1 vanishes in a centre-of-mass frame and is not evaluated
    if (model.C[sh_index(1, 0)] != 0.0 || model.C[sh_index(1, 1)] != 0.0 || model.S[sh_index(1, 1)] != 0.0) {
        std::cerr << "Warning: " << path << " has non-zero degree-1 coefficients; they are ignored." << std::endl;
    }
    model.C[0] = 1.0;    // C00, unused (the point mass is in the EOMs)
    model.S[0] = 0.0;
    for (int n = 2; n <= nmax; n++) {
        for (int m = mmax + 1; m <= n; m++) {
            model.C[sh_index(n, m)] = 0.0;
            model.S[sh_index(n, m)] = 0.0;
        }
    }
    precompute_factors(model);
    return model;
}


namespace {

/**
 * Identity of a file's content for the model cache: size, modification and status-change
 * times (ns), inode. recent is true if either time is within 2 s of now (or in the future):
 * a file rewritten within the clock granularity could then have the same stamp
 * ("racy git" rule), so such a file is loaded but not cached.
 */
struct FileStamp {
    std::uintmax_t size = 0;
    long long mtime_ns = 0;
    long long ctime_ns = 0;
    unsigned long long inode = 0;
    bool recent = true;
};

FileStamp file_stamp(const std::string& path) {
    FileStamp stamp;
#if defined(__unix__) || defined(__APPLE__)
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        return stamp;   // load_spherical_harmonics reports the error
    }
#if defined(__APPLE__)
    const struct timespec mt = st.st_mtimespec, ct = st.st_ctimespec;
#else
    const struct timespec mt = st.st_mtim, ct = st.st_ctim;
#endif
    stamp.size = static_cast<std::uintmax_t>(st.st_size);
    stamp.mtime_ns = static_cast<long long>(mt.tv_sec)*1000000000LL + mt.tv_nsec;
    stamp.ctime_ns = static_cast<long long>(ct.tv_sec)*1000000000LL + ct.tv_nsec;
    stamp.inode = static_cast<unsigned long long>(st.st_ino);
    const std::time_t now = std::time(nullptr);
    stamp.recent = (now - mt.tv_sec < 2) || (now - ct.tv_sec < 2);
#else
    // no inode or status-change time: size and modification time only
    std::error_code ec_size, ec_time;
    stamp.size = std::filesystem::file_size(path, ec_size);
    const auto mtime = std::filesystem::last_write_time(path, ec_time);
    if (ec_size || ec_time) {
        return stamp;
    }
    stamp.mtime_ns = static_cast<long long>(mtime.time_since_epoch().count());
    stamp.recent = (std::filesystem::file_time_type::clock::now() - mtime) < std::chrono::seconds(2);
#endif
    return stamp;
}

}  // namespace


/**
 * Cached load_spherical_harmonics. Thread-safe.
 *
 * The key is (path, nmax, mmax, file size, modification time, status-change time, inode), so an
 * edited, replaced or touched-back file is re-read (on Windows: size and modification time only).
 * A file modified or changed within the last 2 s is loaded but not cached, because a rewrite
 * within the timestamp granularity could keep the same key.
 * Entries are never evicted: each distinct key stays in memory for the lifetime of the process
 * (about 70 MB at nmax = 1200, about 0.3 MB at nmax = 100).
 */
std::shared_ptr<const SphericalHarmonicsModel> load_spherical_harmonics_cached(
    const std::string& path, int nmax, int mmax) {
    static std::mutex cache_mutex;
    using Key = std::tuple<std::string, int, int, std::uintmax_t, long long, long long, unsigned long long>;
    static std::map<Key, std::shared_ptr<const SphericalHarmonicsModel>> cache;
    if (mmax < 0) {
        mmax = -1;
    }
    const FileStamp stamp = file_stamp(path);
    if (stamp.recent) {
        return std::make_shared<const SphericalHarmonicsModel>(load_spherical_harmonics(path, nmax, mmax));
    }
    const Key key = std::make_tuple(path, nmax, mmax, stamp.size, stamp.mtime_ns, stamp.ctime_ns, stamp.inode);
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    auto model = std::make_shared<const SphericalHarmonicsModel>(load_spherical_harmonics(path, nmax, mmax));
    cache[key] = model;
    return model;
}


/**
 * Spherical-harmonics acceleration, gradient and potential in the body-fixed frame
 *
 * Degrees nmin..nmax (the point mass is excluded). Lengths in the unit of R, mu in that
 * unit^3 / time unit^2. The V, W tables are built to degree nmax + 1 for the acceleration
 * and nmax + 2 when the gradient is requested. No allocation.
 *
 * @param model coefficients and factors
 * @param ws V, W tables, sized for model.nmax
 * @param r_bf body-fixed position
 * @param mu gravitational parameter of the coefficients
 * @param R reference radius of the coefficients
 * @param a_bf acceleration to be computed
 * @param G_bf gradient d(a_bf)/d(r_bf) to be computed, or nullptr to skip it
 * @param U potential to be computed, or nullptr to skip it
 * @param rho_min smallest |r_bf|/R evaluated (1 by default: the series may diverge inside the
 *        reference sphere); |r_bf|/R equal to rho_min within a few ulp is accepted
 * @return false if |r_bf|/R < rho_min; outputs are then zero. ws.below_rho_min and ws.rho record the check.
 */
bool sh_accel_bf(const SphericalHarmonicsModel& model,
                 SphericalHarmonicsWorkspace& ws,
                 const double r_bf[3],
                 double mu,
                 double R,
                 double a_bf[3],
                 double G_bf[3][3],
                 double* U,
                 double rho_min) {
    a_bf[0] = a_bf[1] = a_bf[2] = 0.0;
    if (G_bf != nullptr) {
        for (int i = 0; i < 3; i++) {
            G_bf[i][0] = G_bf[i][1] = G_bf[i][2] = 0.0;
        }
    }
    if (U != nullptr) {
        *U = 0.0;
    }

    // position in units of R
    const double xt = r_bf[0] / R;
    const double yt = r_bf[1] / R;
    const double zt = r_bf[2] / R;
    const double rho2 = xt*xt + yt*yt + zt*zt;
    ws.rho = std::sqrt(rho2);
    // 8 eps (relative, on rho^2) accepts a point built as unit vector * rho_min * R
    ws.below_rho_min = !(rho2 >= rho_min*rho_min*(1 - 8*DBL_EPSILON));
    if (ws.below_rho_min) {
        return false;
    }

    // 1. V, W tables to degree N (M&G 3.30-3.31, times N_nm)
    const int nmax = model.nmax;
    const int N = nmax + ((G_bf != nullptr) ? 2 : 1);
    double* V = ws.V.data();
    double* W = ws.W.data();
    const double* rec_a = model.rec_a.data();
    const double* rec_b = model.rec_b.data();
    V[0] = 1.0 / std::sqrt(rho2);
    W[0] = 0.0;
    for (int m = 0; m <= N; m++) {
        const int kmm = sh_index(m, m);
        if (m > 0) {
            const int kp = sh_index(m - 1, m - 1);
            V[kmm] = model.rec_c[m]*(xt*V[kp] - yt*W[kp]) / rho2;
            W[kmm] = model.rec_c[m]*(xt*W[kp] + yt*V[kp]) / rho2;
        }
        if (m + 1 <= N) {
            const int k = sh_index(m + 1, m);
            V[k] = rec_a[k]*zt*V[kmm] / rho2;
            W[k] = rec_a[k]*zt*W[kmm] / rho2;
        }
        for (int n = m + 2; n <= N; n++) {
            const int k = sh_index(n, m);
            const int k1 = sh_index(n - 1, m);
            const int k2 = sh_index(n - 2, m);
            V[k] = (rec_a[k]*zt*V[k1] - rec_b[k]*V[k2]) / rho2;
            W[k] = (rec_a[k]*zt*W[k1] - rec_b[k]*W[k2]) / rho2;
        }
    }

    // 2. accumulate over (n, m)
    double u = 0.0;
    double ax = 0.0, ay = 0.0, az = 0.0;
    double xx = 0.0, yy = 0.0, zz = 0.0, xy = 0.0, xz = 0.0, yz = 0.0;
    for (int n = model.nmin; n <= nmax; n++) {
        const int b1 = sh_index(n + 1, 0);    // start of degree n+1
        const int b2 = sh_index(n + 2, 0);    // start of degree n+2 (used only with G_bf)
        const int mtop = (n < model.mmax) ? n : model.mmax;
        for (int m = 0; m <= mtop; m++) {
            const int k = sh_index(n, m);
            const double c = model.C[k];
            const double s = model.S[k];
            if (c == 0.0 && s == 0.0) {
                continue;
            }
            u += c*V[k] + s*W[k];

            if (m == 0) {
                // S_n0 = 0 and W_n0 = 0
                ax += -model.FAp[k]*c*V[b1 + 1];
                ay += -model.FAp[k]*c*W[b1 + 1];
                az += -model.FAz[k]*c*V[b1];
                if (G_bf != nullptr) {
                    const double v2 = c*V[b2 + 2];
                    const double v0 = c*V[b2];
                    xx +=  0.5*model.FGpp[k]*v2 - 0.5*model.FG0[k]*v0;
                    yy += -0.5*model.FGpp[k]*v2 - 0.5*model.FG0[k]*v0;
                    zz +=  model.FG0[k]*v0;
                    xy +=  0.5*model.FGpp[k]*c*W[b2 + 2];
                    xz +=  model.FGp[k]*c*V[b2 + 1];
                    yz +=  model.FGp[k]*c*W[b2 + 1];
                }
                continue;
            }

            // P(k, j) = c V_kj + s W_kj, Q(k, j) = c W_kj - s V_kj
            const int jp = b1 + m + 1, j0 = b1 + m, jm = b1 + m - 1;
            const double Pp = c*V[jp] + s*W[jp], Qp = c*W[jp] - s*V[jp];
            const double Pm = c*V[jm] + s*W[jm], Qm = c*W[jm] - s*V[jm];
            ax += 0.5*(-model.FAp[k]*Pp + model.FAm[k]*Pm);
            ay += 0.5*(-model.FAp[k]*Qp - model.FAm[k]*Qm);
            az += -model.FAz[k]*(c*V[j0] + s*W[j0]);

            if (G_bf != nullptr) {
                const int i2 = b2 + m;
                double Pmm, Qmm;
                if (m >= 2) {
                    const int imm = i2 - 2;
                    Pmm = c*V[imm] + s*W[imm];
                    Qmm = -(c*W[imm] - s*V[imm]);
                } else {
                    // m = 1: D-^2 E_n1 = -n(n+1) conj(E_{n+2,1})
                    const int i1 = b2 + 1;
                    Pmm = -(c*V[i1] - s*W[i1]);
                    Qmm = -c*W[i1] - s*V[i1];
                }
                const double Ppp = c*V[i2 + 2] + s*W[i2 + 2], Qpp = c*W[i2 + 2] - s*V[i2 + 2];
                const double P1p = c*V[i2 + 1] + s*W[i2 + 1], Q1p = c*W[i2 + 1] - s*V[i2 + 1];
                const double P1m = c*V[i2 - 1] + s*W[i2 - 1], Q1m = c*W[i2 - 1] - s*V[i2 - 1];
                const double P2 = c*V[i2] + s*W[i2];
                const double sym = 0.25*(model.FGpp[k]*Ppp + model.FGmm[k]*Pmm);
                xx +=  sym - 0.5*model.FG0[k]*P2;
                yy += -sym - 0.5*model.FG0[k]*P2;
                zz +=  model.FG0[k]*P2;
                xy +=  0.25*(model.FGpp[k]*Qpp + model.FGmm[k]*Qmm);
                xz +=  0.5*(model.FGp[k]*P1p - model.FGm[k]*P1m);
                yz +=  0.5*(model.FGp[k]*Q1p + model.FGm[k]*Q1m);
            }
        }
    }

    // 3. scale: a ~ mu/R^2, G ~ mu/R^3, U ~ mu/R
    const double ka = mu / (R*R);
    a_bf[0] = ka*ax;
    a_bf[1] = ka*ay;
    a_bf[2] = ka*az;
    if (G_bf != nullptr) {
        const double kg = mu / (R*R*R);
        G_bf[0][0] = kg*xx;  G_bf[0][1] = kg*xy;  G_bf[0][2] = kg*xz;
        G_bf[1][0] = kg*xy;  G_bf[1][1] = kg*yy;  G_bf[1][2] = kg*yz;
        G_bf[2][0] = kg*xz;  G_bf[2][1] = kg*yz;  G_bf[2][2] = kg*zz;
    }
    if (U != nullptr) {
        *U = mu / R * u;
    }
    return true;
}


/**
 * Spherical-harmonics perturbation in the frame of integration
 *
 * With T = pxform(frame, frame_bf, et), r_bf = T r, a = T^T a_bf and GG = T^T G_bf T.
 * The rotation rate does not enter: GG is d(a)/d(r) at fixed t.
 *
 * @param rvec spacecraft position in the frame of integration, in DU
 * @param et epoch, in ephemeris seconds
 * @param frame_cspice NAIF name of frame of integration
 * @param frame_bf_cspice NAIF name of the body-fixed frame of the coefficients
 * @param model coefficients and factors
 * @param ws V, W tables, sized for model.nmax
 * @param mu_sh GM of the coefficients, scaled for DU, TU (GM_file TU^2 / DU^3)
 * @param R_sh reference radius of the coefficients, divided by DU
 * @param a acceleration to be computed, in DU/TU^2
 * @param GG 3-by-3 gradient to be computed, in 1/TU^2, or nullptr to skip it
 * @param rho_min smallest |r|/R evaluated, see sh_accel_bf
 * @return false on a SPICE error (failed_c() is left set) or if |r|/R < rho_min (failed_c() is not set)
 */
bool sh_perturbation(const double rvec[3],
                     double et,
                     const char* frame_cspice,
                     const char* frame_bf_cspice,
                     const SphericalHarmonicsModel& model,
                     SphericalHarmonicsWorkspace& ws,
                     double mu_sh,
                     double R_sh,
                     double a[3],
                     double GG[3][3],
                     double rho_min) {
    ws.below_rho_min = false;
    SpiceDouble T[3][3];
    pxform_c(frame_cspice, frame_bf_cspice, et, T);
    if (failed_c()) {
        return false;
    }

    double r_bf[3];
    for (int i = 0; i < 3; i++) {
        r_bf[i] = T[i][0]*rvec[0] + T[i][1]*rvec[1] + T[i][2]*rvec[2];
    }
    double a_bf[3];
    double G_bf[3][3];
    if (!sh_accel_bf(model, ws, r_bf, mu_sh, R_sh, a_bf, (GG != nullptr) ? G_bf : nullptr, nullptr, rho_min)) {
        return false;
    }

    // a = T^T a_bf
    for (int i = 0; i < 3; i++) {
        a[i] = T[0][i]*a_bf[0] + T[1][i]*a_bf[1] + T[2][i]*a_bf[2];
    }
    if (GG != nullptr) {
        // GG = T^T (G_bf T)
        double GT[3][3];
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                GT[j][k] = G_bf[j][0]*T[0][k] + G_bf[j][1]*T[1][k] + G_bf[j][2]*T[2][k];
            }
        }
        for (int i = 0; i < 3; i++) {
            for (int k = 0; k < 3; k++) {
                GG[i][k] = T[0][i]*GT[0][k] + T[1][i]*GT[1][k] + T[2][i]*GT[2][k];
            }
        }
    }
    return true;
}
