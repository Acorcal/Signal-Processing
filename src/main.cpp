#include <matio.h>
#include <armadillo>

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

static void ensure(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

// List top-level variables (PUBLIC API)
static void listTopLevelVars(mat_t* mat) {
    Mat_Rewind(mat);
    matvar_t* info = nullptr;

    std::cout << "Top-level variables:\n";
    while ((info = Mat_VarReadNextInfo(mat)) != nullptr) {
        std::cout << "  - " << (info->name ? info->name : "<noname>")
                  << " | class=" << info->class_type
                  << " type=" << info->data_type
                  << " rank=" << info->rank;

        if (info->rank >= 1 && info->dims) {
            std::cout << " dims=";
            for (int i = 0; i < info->rank; ++i) {
                std::cout << info->dims[i] << (i + 1 < info->rank ? "x" : "");
            }
        }
        std::cout << "\n";

        Mat_VarFree(info);
    }

    Mat_Rewind(mat);
    std::cout << "\n";
}

// Read TD160.data as a real double rank-2 matrix into Armadillo
static arma::mat readStructFieldDoubleMat(mat_t* mat,
                                         const std::string& structName,
                                         const std::string& fieldName,
                                         size_t structIndex = 0)
{
    matvar_t* s = Mat_VarRead(mat, structName.c_str());
    ensure(s != nullptr, "Struct not found: " + structName);
    ensure(s->class_type == MAT_C_STRUCT, "'" + structName + "' is not a struct.");

    matvar_t* f = Mat_VarGetStructFieldByName(s, fieldName.c_str(), structIndex);
    ensure(f != nullptr, "Field not found: " + structName + "." + fieldName);

    ensure(f->class_type == MAT_C_DOUBLE && f->data_type == MAT_T_DOUBLE,
           "Field '" + structName + "." + fieldName + "' is not double.");
    ensure(f->isComplex == 0, "Field is complex; not handled.");
    ensure(f->rank == 2, "Field is not rank-2.");
    ensure(f->data != nullptr, "Field has no data.");

    const arma::uword rows = (arma::uword)f->dims[0];
    const arma::uword cols = (arma::uword)f->dims[1];
    const size_t n_elems = (f->nbytes / f->data_size);

    arma::mat A(rows, cols);
    std::memcpy(A.memptr(), f->data, n_elems * sizeof(double));

    Mat_VarFree(s);  // IMPORTANT
    return A;
}

int main(int argc, char** argv) {
    // Usage:
    //   ./hello_cpp.exe <mat_path> [structName] [fieldName] [channelIndex]
    const std::string matPath    = (argc >= 2) ? argv[1] : "TD160.mat";
    const std::string structName = (argc >= 3) ? argv[2] : "TD160";
    const std::string fieldName  = (argc >= 4) ? argv[3] : "data";
    const arma::uword ch         = (argc >= 5) ? (arma::uword)std::stoul(argv[4]) : 0;

    mat_t* mat = Mat_Open(matPath.c_str(), MAT_ACC_RDONLY);
    if (!mat) {
        std::cerr << "Could not open MAT file: " << matPath << "\n";
        return 1;
    }

    try {
        std::cout << "Opened: " << matPath << "\n\n";
        listTopLevelVars(mat);

        arma::mat data = readStructFieldDoubleMat(mat, structName, fieldName);

        std::cout << "Loaded " << structName << "." << fieldName
                  << " as " << data.n_rows << " x " << data.n_cols << "\n";

        // Assume samples x channels (common). Choose a channel (column).
        ensure(ch < data.n_cols, "Channel out of range (cols=" + std::to_string(data.n_cols) + ").");

        arma::vec y = data.col(ch);

        std::cout << "Using channel " << ch << ", y has n=" << y.n_elem << "\n";
        std::cout << "First 10 samples:\n";
        for (arma::uword i = 0; i < std::min<arma::uword>(10, y.n_elem); ++i) {
            std::cout << "  y[" << i << "] = " << y(i) << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        Mat_Close(mat);
        return 1;
    }

    Mat_Close(mat);
    return 0;
}
