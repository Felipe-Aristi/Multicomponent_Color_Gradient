#ifndef SAVE_DATA_CUH
#define SAVE_DATA_CUH

#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <stdexcept>

#include "../constants.cuh"
#include "../utilities/types.cuh"
#include "../utilities/cudaUtilities.cuh"
#include "../memory.cuh"

// Folder: <cwd>/LBM_bubble/vtk
inline std::filesystem::path default_out_dir()
{
    return std::filesystem::current_path() / "LBM_bubble" / "vtk";
}

inline void write_vtk_legacy(const std::filesystem::path &filename,
                             const real_t *rhor,
                             const real_t *rhob,
                             const real_t *ux,
                             const real_t *uy,
                             const real_t *uz,
                             bool write_total_rho = true)
{
    std::ofstream out(filename);
    if (!out)
        throw std::runtime_error("Cannot open VTK file for writing: " + filename.string());

    out << "# vtk DataFile Version 3.0\n";
    out << "LBM output\n";
    out << "ASCII\n";
    out << "DATASET STRUCTURED_POINTS\n";
    out << "DIMENSIONS " << NX << " " << NY << " " << NZ << "\n";
    out << "ORIGIN 0 0 0\n";
    out << "SPACING 1 1 1\n";
    out << "POINT_DATA " << Ncells << "\n";

    out << std::setprecision(16);

    // rhor
    out << "SCALARS rhor double 1\n";
    out << "LOOKUP_TABLE default\n";
    for (std::size_t id = 0; id < Ncells; ++id)
        out << (double)rhor[id] << "\n";

    // rhob
    out << "SCALARS rhob double 1\n";
    out << "LOOKUP_TABLE default\n";
    for (std::size_t id = 0; id < Ncells; ++id)
        out << (double)rhob[id] << "\n";

    // rho_total (optional)
    if (write_total_rho)
    {
        out << "SCALARS rho double 1\n";
        out << "LOOKUP_TABLE default\n";
        for (std::size_t id = 0; id < Ncells; ++id)
            out << (double)(rhor[id] + rhob[id]) << "\n";
    }

    // velocity vector
    out << "VECTORS u double\n";
    for (std::size_t id = 0; id < Ncells; ++id)
        out << (double)ux[id] << " " << (double)uy[id] << " " << (double)uz[id] << "\n";
}

// Copies D->H and writes one file for "step"
inline void write_vtk_step_device(int step, const LbmDevice &d, LbmHost &h)
{
    // Ensure previous kernels are finished before copying
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy rhor, rhob, ux, uy, uz from device to pinned host buffers
    copy_out_D2H(h, d);

    const auto out_dir = default_out_dir();
    std::filesystem::create_directories(out_dir);

    std::ostringstream name;
    name << "lbm_" << std::setw(8) << std::setfill('0') << step << ".vtk";

    write_vtk_legacy(out_dir / name.str(), h.rhor, h.rhob, h.ux, h.uy, h.uz, true);
}

#endif