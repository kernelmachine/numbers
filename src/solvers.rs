use super::{Matrix, Trans};
use lapack::*;
use factorizations::*;
use matrixerror::MatrixError;



/// Solve Ax = b via LU Factorization.
pub fn lusolve(a : &mut Matrix<f64>, b : &mut Matrix<f64>) ->  Result<Matrix<f64>,MatrixError>{
    let (a,ipiv) = try!(lufact(a));
    let lda = a.row_size;
    let n = a.col_size;
    let ldb = b.row_size;
    let nrhs = b.col_size;
    let mut info = 0;
    
    dgetrs(b'N', n, nrhs, &a.elements, lda, &ipiv, &mut b.elements, ldb , &mut info);

    match info {
        x if x > 0 => Err(MatrixError::LapackComputationError),
        0 => Ok(Matrix {
            elements : b.elements.to_owned(),
            row_size : ldb,
            col_size : nrhs,
            transpose : Trans :: Regular,
        }),
        x if x < 0 => Err(MatrixError::LapackInputError),
        _ => Err(MatrixError::UnknownError)
    }


}

/// Solve Ax = b via least squares error solution.

pub fn linearsolve(){
    unimplemented!();
}

/// Solve Ax = b via SVD.
pub fn svdsolve(){
    unimplemented!();
}

/// Solve Ax = b via Cholesky decomposition.
pub fn cholsolve(){
    unimplemented!();
}
