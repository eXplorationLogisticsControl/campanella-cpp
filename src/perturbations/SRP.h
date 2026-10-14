#ifndef SRP_H
#define SRP_H

#include <vector>
#include <SpiceUsr.h>          // access SPICE files


std::vector<double> compute_SRP(double x,
                                                   double y,
                                                   double z,
                                                   SpiceDouble et,
                                                   ConstSpiceChar * frame_cspice,
                                                   ConstSpiceChar * ref_body_name,
                                                   double A,
                                                   double m,
                                                   double rhoS,
                                                   double rhoD,
                                                   double TU,
                                                   double LU);

#endif 