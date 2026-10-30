/**
 * SpiceParameters: resolution and validation of the dynamics model
 */

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <SpiceUsr.h>          // access SPICE files

#include "campanella.hpp"
#include "model.h"

namespace campanella {
namespace detail {

Workspace::Workspace(const Model& model) : body_pos(3*model.codes.size(), 0.0) {
    if (model.sh) {
        sh_ws.emplace(model.sh->nmax);
    }
}

namespace {

constexpr int NAIF_SUN = 10;
constexpr SpiceInt INERTIAL_FRAME_CLASS = 1;

[[noreturn]] void invalid(const std::string& what) {
    throw std::invalid_argument("SpiceParameters: " + what);
}

int resolve_naif_code(const std::string& name) {
    SpiceInt code = 0;
    SpiceBoolean found = SPICEFALSE;
    bods2c_c(name.c_str(), &code, &found);
    spice_throw_if_failed();
    if (!found) {
        invalid("unknown NAIF body \"" + name + "\"");
    }
    return static_cast<int>(code);
}

void check_inertial_frame(const std::string& frame, const char* what) {
    SpiceInt frcode = 0;
    namfrm_c(frame.c_str(), &frcode);
    spice_throw_if_failed();
    if (frcode == 0) {
        invalid(std::string("unknown ") + what + " \"" + frame + "\"");
    }
    SpiceInt cent, frclss, clssid;
    SpiceBoolean found = SPICEFALSE;
    frinfo_c(frcode, &cent, &frclss, &clssid, &found);
    spice_throw_if_failed();
    if (!found || frclss != INERTIAL_FRAME_CLASS) {
        invalid(std::string(what) + " \"" + frame + "\" is not inertial");
    }
}

double gm_from_kernels(int code, const std::string& name) {
    if (!bodfnd_c(code, "GM")) {
        invalid("no GM for body \"" + name + "\" (" + std::to_string(code) + ") in the kernel pool; "
                "load a PCK such as gm_de440.tpc, or pass GMs");
    }
    SpiceInt dim = 0;
    SpiceDouble gm = 0.0;
    bodvcd_c(code, "GM", 1, &dim, &gm);
    spice_throw_if_failed();
    return gm;
}

std::shared_ptr<const Model> build_model(double et0, std::vector<std::string> naif_ids,
                                         SpiceOptions options) {
    init_spice();
    auto model = std::make_shared<Model>();
    if (!std::isfinite(et0)) {
        invalid("et0 must be finite");
    }
    if (naif_ids.empty()) {
        invalid("naif_ids must contain at least the central body");
    }
    std::string abcorr = options.abcorr;
    std::transform(abcorr.begin(), abcorr.end(), abcorr.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    if (abcorr != "NONE") {
        invalid("abcorr must be \"NONE\": the force model uses geometric positions");
    }

    // bodies and frame (built-in names and frames need no kernels)
    std::vector<int> codes;
    for (const std::string& name : naif_ids) {
        int code = resolve_naif_code(name);
        if (std::find(codes.begin(), codes.end(), code) != codes.end()) {
            invalid("NAIF body \"" + name + "\" (" + std::to_string(code) + ") is listed twice");
        }
        codes.push_back(code);
    }
    check_inertial_frame(options.naif_frame, "naif_frame");

    // gravitational parameters, km^3/s^2
    if (options.GMs.empty()) {
        for (size_t i = 0; i < codes.size(); i++) {
            options.GMs.push_back(gm_from_kernels(codes[i], naif_ids[i]));
        }
    } else if (options.GMs.size() != codes.size()) {
        invalid("GMs must have one entry per NAIF id");
    }
    for (size_t i = 0; i < codes.size(); i++) {
        const double gm = options.GMs[i];
        if (!std::isfinite(gm) || gm < 0.0 || (i == 0 && gm == 0.0)) {
            invalid("GMs must be finite and >= 0, and GMs[0] > 0");
        }
    }

    // units
    double DU = 1.0, TU = 1.0, VU = 1.0;
    if (options.DU && !(std::isfinite(*options.DU) && *options.DU > 0.0)) {
        invalid("DU must be finite and > 0");
    }
    const bool canonical = options.DU.has_value() && options.use_canonical_scales;
    if (canonical) {
        DU = *options.DU;
        VU = std::sqrt(options.GMs[0] / DU);
        TU = DU / VU;
    }
    std::vector<double> mus;
    for (double gm : options.GMs) {
        mus.push_back(canonical ? gm / options.GMs[0] : gm);
    }

    // solar radiation pressure
    int idx_sun = -1;
    if (options.include_srp) {
        auto it = std::find(codes.begin(), codes.end(), NAIF_SUN);
        if (it == codes.end()) {
            invalid("include_srp requires the Sun (\"10\" or \"SUN\") in naif_ids");
        }
        idx_sun = static_cast<int>(it - codes.begin());
        if (!(std::isfinite(options.srp_A) && options.srp_A >= 0.0)) {
            invalid("srp_A must be finite and >= 0");
        }
        if (!(std::isfinite(options.srp_m) && options.srp_m > 0.0)) {
            invalid("srp_m must be finite and > 0");
        }
        if (!(std::isfinite(options.srp_rhoS) && options.srp_rhoS >= 0.0
              && std::isfinite(options.srp_rhoD) && options.srp_rhoD >= 0.0)) {
            invalid("srp_rhoS and srp_rhoD must be finite and >= 0");
        }
    }

    // central-body spherical harmonics
    if (!options.sh_file.empty()) {
        if (options.sh_nmax < 2) {
            invalid("sh_nmax must be >= 2 when sh_file is given");
        }
        if (!(options.sh_rho_min > 0.0 && options.sh_rho_min <= 1.0)) {
            invalid("sh_rho_min must be in (0, 1]");
        }
        if (options.sh_frame.empty()) {
            invalid("sh_frame (e.g. MOON_PA, ITRF93) is required with sh_file");
        }
        SpiceInt frcode = 0;
        namfrm_c(options.sh_frame.c_str(), &frcode);
        spice_throw_if_failed();
        if (frcode == 0) {
            invalid("unknown sh_frame \"" + options.sh_frame + "\" (are its frame kernels loaded?)");
        }
        std::shared_ptr<const SphericalHarmonicsModel> sh =
            load_spherical_harmonics_cached(options.sh_file, options.sh_nmax, options.sh_mmax);
        // the point mass uses GMs[0], the series the GM of the file: they must be the same body
        if (!(std::abs(sh->GM_km3s2 - options.GMs[0]) <= 1e-5*sh->GM_km3s2)) {
            std::ostringstream msg;
            msg.precision(16);
            msg << "GMs[0] = " << options.GMs[0] << " km^3/s^2 differs from the GM of " << options.sh_file
                << " (" << sh->GM_km3s2 << " km^3/s^2) by more than 1e-5";
            invalid(msg.str());
        }
        model->sh = sh;
        model->mu_sh = sh->GM_km3s2*TU*TU / (DU*DU*DU);
        model->R_sh = sh->R_km / DU;
    }

    model->et0 = et0;
    model->naif_ids = std::move(naif_ids);
    model->codes = std::move(codes);
    model->mus = std::move(mus);
    model->DU = DU;
    model->TU = TU;
    model->VU = VU;
    model->idx_sun = idx_sun;
    model->options = std::move(options);
    return model;
}

}  // namespace
}  // namespace detail


SpiceParameters::SpiceParameters(double et0, std::vector<std::string> naif_ids, SpiceOptions options)
    : model_(detail::build_model(et0, std::move(naif_ids), std::move(options))),
      ws_(std::make_unique<detail::Workspace>(*model_)) {}

SpiceParameters::SpiceParameters(const SpiceParameters& other)
    : model_(other.model_), ws_(std::make_unique<detail::Workspace>(*other.model_)) {}

SpiceParameters& SpiceParameters::operator=(const SpiceParameters& other) {
    if (this != &other) {
        ws_ = std::make_unique<detail::Workspace>(*other.model_);
        model_ = other.model_;
    }
    return *this;
}

SpiceParameters::~SpiceParameters() = default;

double SpiceParameters::et0() const { return model_->et0; }
const std::vector<std::string>& SpiceParameters::naif_ids() const { return model_->naif_ids; }
const std::vector<int>& SpiceParameters::naif_codes() const { return model_->codes; }
const std::string& SpiceParameters::naif_frame() const { return model_->options.naif_frame; }
const std::vector<double>& SpiceParameters::GMs() const { return model_->options.GMs; }
const std::vector<double>& SpiceParameters::mus() const { return model_->mus; }
double SpiceParameters::DU() const { return model_->DU; }
double SpiceParameters::TU() const { return model_->TU; }
double SpiceParameters::VU() const { return model_->VU; }
bool SpiceParameters::include_srp() const { return model_->idx_sun >= 0; }
bool SpiceParameters::include_spherical_harmonics() const { return model_->sh != nullptr; }
const SpiceOptions& SpiceParameters::options() const { return model_->options; }

}  // namespace campanella
