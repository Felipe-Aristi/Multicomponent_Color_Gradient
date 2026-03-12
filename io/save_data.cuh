#ifndef SAVE_DATA_CUH
#define SAVE_DATA_CUH

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "../constants.cuh"
#include "../utilities/types.cuh"
#include "../utilities/cudaUtilities.cuh"
#include "../memory.cuh"

// Folder: <cwd>/LBM_bubble/vti
inline std::filesystem::path default_out_dir()
{
    return std::filesystem::current_path() / "JET_VTK" / "vti";
}

inline const char *vtk_real_type()
{
    if constexpr (std::is_same_v<real_t, float>)
        return "Float32";
    else
        return "Float64";
}

inline void write_uint64_block_header(std::ofstream &out, std::uint64_t nbytes)
{
    out.write(reinterpret_cast<const char *>(&nbytes), sizeof(std::uint64_t));
    if (!out)
        throw std::runtime_error("Failed writing VTI block header.");
}

inline void write_raw_block(std::ofstream &out, const real_t *data, std::size_t nvals)
{
    const std::uint64_t nbytes =
        static_cast<std::uint64_t>(nvals) * static_cast<std::uint64_t>(sizeof(real_t));

    write_uint64_block_header(out, nbytes);
    out.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(nbytes));

    if (!out)
        throw std::runtime_error("Failed writing VTI raw block.");
}

inline void write_sum_block(std::ofstream &out,
                            const real_t *a,
                            const real_t *b,
                            std::size_t nvals)
{
    const std::uint64_t nbytes =
        static_cast<std::uint64_t>(nvals) * static_cast<std::uint64_t>(sizeof(real_t));

    write_uint64_block_header(out, nbytes);

    std::vector<real_t> tmp(nvals);
    for (std::size_t i = 0; i < nvals; ++i)
        tmp[i] = a[i] + b[i];

    out.write(reinterpret_cast<const char *>(tmp.data()), static_cast<std::streamsize>(nbytes));

    if (!out)
        throw std::runtime_error("Failed writing VTI sum block.");
}

inline void write_vec3_block(std::ofstream &out,
                             const real_t *ux,
                             const real_t *uy,
                             const real_t *uz,
                             std::size_t npts)
{
    const std::uint64_t nbytes =
        static_cast<std::uint64_t>(3) *
        static_cast<std::uint64_t>(npts) *
        static_cast<std::uint64_t>(sizeof(real_t));

    write_uint64_block_header(out, nbytes);

    std::vector<real_t> tmp(3 * npts);
    for (std::size_t i = 0; i < npts; ++i)
    {
        tmp[3 * i + 0] = ux[i];
        tmp[3 * i + 1] = uy[i];
        tmp[3 * i + 2] = uz[i];
    }

    out.write(reinterpret_cast<const char *>(tmp.data()), static_cast<std::streamsize>(nbytes));

    if (!out)
        throw std::runtime_error("Failed writing VTI vector block.");
}

inline void write_vti(const std::filesystem::path &filename,
                      const real_t *rhor,
                      const real_t *rhob,
                      const real_t *ux,
                      const real_t *uy,
                      const real_t *uz,
                      bool write_total_rho = true)
{
    std::ofstream out(filename, std::ios::binary);
    if (!out)
        throw std::runtime_error("Cannot open VTI file for writing: " + filename.string());

    constexpr std::size_t npts = static_cast<std::size_t>(Ncells);
    constexpr std::uint64_t header_bytes = sizeof(std::uint64_t);

    const std::uint64_t bytes_scalar =
        header_bytes + static_cast<std::uint64_t>(npts) * sizeof(real_t);

    const std::uint64_t bytes_vec3 =
        header_bytes + static_cast<std::uint64_t>(3) * static_cast<std::uint64_t>(npts) * sizeof(real_t);

    std::uint64_t off_rhor = 0;
    std::uint64_t off_rhob = off_rhor + bytes_scalar;
    std::uint64_t off_rho = off_rhob + bytes_scalar;
    std::uint64_t off_u = off_rho + (write_total_rho ? bytes_scalar : 0);

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
    out << "  <ImageData WholeExtent=\"0 " << (NX - 1)
        << " 0 " << (NY - 1)
        << " 0 " << (NZ - 1)
        << "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n";
    out << "    <Piece Extent=\"0 " << (NX - 1)
        << " 0 " << (NY - 1)
        << " 0 " << (NZ - 1) << "\">\n";

    out << "      <PointData Scalars=\"rhor\" Vectors=\"u\">\n";
    out << "        <DataArray type=\"" << vtk_real_type()
        << "\" Name=\"rhor\" format=\"appended\" offset=\"" << off_rhor << "\"/>\n";
    out << "        <DataArray type=\"" << vtk_real_type()
        << "\" Name=\"rhob\" format=\"appended\" offset=\"" << off_rhob << "\"/>\n";

    if (write_total_rho)
    {
        out << "        <DataArray type=\"" << vtk_real_type()
            << "\" Name=\"rho\" format=\"appended\" offset=\"" << off_rho << "\"/>\n";
    }

    out << "        <DataArray type=\"" << vtk_real_type()
        << "\" Name=\"u\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << off_u << "\"/>\n";
    out << "      </PointData>\n";
    out << "      <CellData>\n";
    out << "      </CellData>\n";
    out << "    </Piece>\n";
    out << "  </ImageData>\n";
    out << "  <AppendedData encoding=\"raw\">\n";
    out << "_";

    // appended raw blocks
    write_raw_block(out, rhor, npts);
    write_raw_block(out, rhob, npts);

    if (write_total_rho)
        write_sum_block(out, rhor, rhob, npts);

    write_vec3_block(out, ux, uy, uz, npts);

    out << "\n  </AppendedData>\n";
    out << "</VTKFile>\n";

    if (!out)
        throw std::runtime_error("Error while finalizing VTI file: " + filename.string());
}

// Copies D->H and writes one file for "step"
inline void write_vti_step_device(int step, const LbmDevice &d, LbmHost &h)
{
    CUDA_CHECK(cudaDeviceSynchronize());
    copy_out_D2H(h, d);

    const auto out_dir = default_out_dir();
    std::filesystem::create_directories(out_dir);

    std::ostringstream name;
    name << "lbm_" << std::setw(8) << std::setfill('0') << step << ".vti";

    write_vti(out_dir / name.str(), h.rhor, h.rhob, h.ux, h.uy, h.uz, true);
}

#endif