use super::{Matrix, SVD};
use std::cmp::{min,max};
use lapack::*;
use matrixerror::MatrixError;
use scalars::*;

/// Compute the LU factorization.
pub fn lu(a : &mut Matrix<f64>) -> Result<(&mut Matrix<f64>, Vec<i32>), MatrixError>{
    let m = a.row_size;
    let n = a.col_size;
    let mut ipiv = vec![0; min(m,n)];
    let mut info = 0;
    dgetrf(m, n, &mut a.elements, m, &mut ipiv, &mut info);
    match info {
        1 => Err(MatrixError::LapackComputationError),
        0 => Ok((a, ipiv)),
        -1 => Err(MatrixError::LapackInputError),
        _ => Err(MatrixError::UnknownError)
    }


}


/// Compute the QR Factorization.
pub fn qr(a : &mut Matrix<f64>) ->Result<Matrix<f64>,MatrixError>{
    let m = a.row_size;
    let n = a.col_size;
    let mut tau = vec![0.0; min(m,n)];
    let mut work = vec![0.0; 4*n];
    let lwork = 4*n as isize;
    let mut info = 0;
    dgeqrf(m, n, &mut a.elements, m, &mut tau,
    &mut work, lwork, &mut info);
    match info {
        1 => Err(MatrixError::LapackComputationError),
        0 => Ok(Matrix {
            elements : a.elements.to_owned(),
            row_size : m,
            col_size : n,
            transpose : false
        }),
        -1 => Err(MatrixError::LapackInputError),
        _ => Err(MatrixError::UnknownError)
    }
}

/// Compute the SVD Factorization.
pub fn svd(a : &mut Matrix<f64>) -> Result <SVD, MatrixError> {
    let m = a.row_size;
    let n = a.col_size;

    let lda = a.row_size;

    let s = singular_values(a);
    let ldu = a.row_size;
    let mut u = vec![0.0; ldu*min(m,n)];

    let ldvt = a.col_size;
    let mut vt = vec![0.0;ldvt*n];

    let lwork = max(max(1,3*min(m,n)+min(m,n)),5*min(m,n)) +10 ;
    let mut work = vec![0.0; lwork];

    let mut info = 0;
    let mut s_elem = s.unwrap().elements;
    dgesvd(b'S', b'S',m,n,&mut a.elements,lda,&mut s_elem, &mut u,ldu, &mut vt, ldvt, &mut work, lwork as isize, &mut info);

    match info {
        1 => Err(MatrixError::LapackComputationError),
        0 => Ok((
            Matrix {
                elements : u,
                row_size : ldu,
                col_size : min(m,n),
                transpose : false,
            },
            Matrix :: diag_mat(s_elem)
            ,
            Matrix {
                elements : vt,
                row_size : ldvt,
                col_size : n,
                transpose : true,
            }
        )

),
        -1 => Err(MatrixError::LapackInputError),
        _ => Err(MatrixError::UnknownError)
    }



}


/// Compute the Cholesky Factorization.
pub fn cholesky(){
    unimplemented!();
}

/// Compute the Hessenberg Factorization.
pub fn hess(){
    unimplemented!();


}

/// Compute the Schur Factorization.
pub fn schur(){
    unimplemented!();

}
