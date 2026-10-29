#ifndef THIRDBODY_H
#define THIRDBODY_H

#include <vector>
#include <SpiceUsr.h>          // access SPICE files


std::vector<double> compute_thirdbody_perturbation(double x,
                                                   double y,
                                                   double z,
                                                   double GM_i,
                                                   ConstSpiceChar * third_body_name,
                                                   SpiceDouble et,
                                                   ConstSpiceChar * frame_cspice,
                                                   ConstSpiceChar * ref_body_name,
                                                   double LU);

#endif 