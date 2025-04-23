#define XDIM 128
#define YDIM 128
#define ZDIM 128
//constexpr for avoiding multiple definitions of constants
constexpr double alpha = 0.05; //buoyancy coefficient (0.05 – 0.1)
constexpr double beta = 0.1; //temperature lift coefficient
constexpr double T_ambient = 0.0; //Ambient temperature
const double dx = 1.0 / (XDIM - 1); // Grid spacing in x-axis
const double dy = 1.0 / (YDIM - 1); // Grid spacing in y-axis
const double dz = 1.0 / (ZDIM - 1); // Grid spacing in z-axis
const double dt = 0.5; //Time increment

// The maximum number of iterations for the Conjugate Gradient algorithm.
const int cg_max_iterations = 100; 

// The convergence criterion. The algorithm will stop if the residual becomes
// smaller than this value.
const float cg_tolerance = 1e-5f; 