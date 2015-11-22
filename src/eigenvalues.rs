use super::{Matrix, Eig, Triangular, Trans};
use lapack::*;
use matrixerror::MatrixError;



/// Get the eigenvalues of a matrix.
pub fn eigenvalues(a : &mut Matrix<f64>, eorv : Eig, tri : Triangular) -> Result<Matrix<f64>,MatrixError>{
    let n = a.col_size;
    let mut w = vec![0.0; n];
    let mut work = vec![0.0; 4 * n];
    let lwork = 4 * n as isize;
    let mut info = 0;
    let e = match eorv {
        Eig::Eigenvalues => b'V',
        Eig::EigenvaluesAndEigenvectors => b'E'
    };

    let t = match tri {
        Triangular::Upper => b'U',
        Triangular::Lower => b'L'
    };

    dsyev(e, t, n, &mut a.elements, n, &mut w, &mut work, lwork, &mut info);
    
    match info {
        1 => Err(MatrixError::LapackComputationError),
        0 => Ok (Matrix {
            elements : w.to_owned(),
            row_size : w.len(),
            col_size : 1,
            transpose : Trans :: Regular,
        }),
        -1 => Err(MatrixError::LapackInputError),
        _ => Err(MatrixError::UnknownError)
    }

}

/// Get the eigenvalues of a hermitian matrix.
pub fn eigsh() -> Result<Matrix<f64>,MatrixError> {
    unimplemented!();
}
