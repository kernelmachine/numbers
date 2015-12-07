use super::{Matrix, Eig, SVD, Triangular, Trans, RectMat, SqMat};
use std::cmp::{min,max};
use lapack::*;
use matrixerror::MatrixError;
use eigenvalues::*;
use operations::*;

/// Compute the LU factorization.
///
/// # Arguments
///
/// * `Matrix` - Matrix of type f64
///
/// # Example
/// ```
/// #[macro_use] extern crate numbers;
/// use numbers::Matrix;
/// pub fn main(){
/// let mut mat  = Matrix :: new(vec![6.0, 0.0 ,2.0, 24.0, 1.0, 8.0, -12.0, 1.0 ,-3.0],3,3).ok().unwrap();
/// let (mut a,ipiv) = numbers::factorizations::lufact(&mut mat).ok().unwrap();
/// let mut l = numbers::operations::triu(&mut a,0).ok().unwrap();
/// let mut u = numbers::operations::tril(&mut a, 0).ok().unwrap();
/// matrix_equal!(numbers::operations::dot(&mut l, &mut u).ok().unwrap(), a)
///}

pub fn lufact<T: SqMat> (a : &mut T) -> Result<(&mut T, Vec<i32>), MatrixError>{
    let m = a.row_size;
    let n = a.col_size;
    let mut ipiv = vec![0; min(m,n)];
    let mut info = 0;
    dgetrf(m, n, &mut a.elements, m, &mut ipiv, &mut info);
    match info {
        x if x > 0 => Err(MatrixError::LapackComputationError),
        0 => Ok((a, ipiv)),
        x if x < 0 => Err(MatrixError::LapackInputError),
        _ => Err(MatrixError::UnknownError)
    }


}



/// Compute the SVD Factorization.
///
/// # Arguments
///
/// * `Matrix` - Matrix of type f64
///
/// # Example
/// ```
/// #[macro_use] extern crate numbers;
/// use numbers::Matrix;
/// pub fn main(){
/// let mut mat  = Matrix :: new(vec![1.0,1.0,2.0,2.0,4.0,4.0, 8.0,8.0, 10.0],3,3).ok().unwrap();
/// let (mut u, mut e, mut v) = numbers::factorizations::svd(&mut mat).ok().unwrap();
/// let mut ue = numbers::operations::dot(&mut u,&mut e).ok().unwrap();
/// let uev = numbers::operations::dot(&mut ue,&mut v).ok().unwrap();
/// matrix_equal!(uev, mat)
///}

pub fn svd <T: SqMat> (a : &mut T) -> Result <SVD, MatrixError> {
    let m = a.row_size;
    let n = a.col_size;
    let lda = a.row_size;
    let s = singular_values(a);
    let ldu = a.row_size;
    let ldvt = a.col_size;

    let mut u = vec![0.0; ldu*m];
    let mut vt = vec![0.0;ldvt*n];

    let lwork = max(max(1,3*min(m,n)+min(m,n)),5*min(m,n));
    let mut work = vec![0.0; lwork];
    let mut info = 0;

    if let Ok(mut s) = singular_values(a){
        dgesvd(b'A', b'A',m,n,&mut a.elements,lda,&mut s.elements, &mut u,ldu, &mut vt,
        ldvt, &mut work, lwork as isize, &mut info);

        match info {
            x if x > 0 => return Err(MatrixError::LapackComputationError),
            0 => return Ok((
                Matrix {
                    elements : u,
                    row_size : ldu,
                    col_size : m,
                    transpose : Trans :: Regular,
                },
                Matrix :: diag_mat(s.elements),
                Matrix {
                    elements : vt,
                    row_size : ldvt,
                    col_size : n,
                    transpose : Trans :: Transpose,
                }
            )

    ),
            x if x < 0 => return Err(MatrixError::LapackInputError),
            _ => return Err(MatrixError::UnknownError)
        }

    }
    Err(MatrixError::LapackComputationError)


}


/// Compute the singular values of a matrix.
///
/// # Arguments
///
/// * `Matrix` - Matrix of type f64
///
/// # Example
/// ```
/// #[macro_use] extern crate numbers;
/// use numbers::Matrix;
/// pub fn main(){
/// let mut a  = Matrix :: new(vec![1.0,1.0,2.0,2.0,4.0,4.0, 8.0,8.0, 10.0],3,3).ok().unwrap();
/// let s = numbers::factorizations::singular_values(&mut a).ok().unwrap();
/// let ans = Matrix :: new(vec![16.0,  1.0, 0.5  ], 3, 1).ok().unwrap();
/// matrix_equal!(s, ans)
///}
pub fn singular_values <T: SqMat> (a : &mut T) -> Result<T, MatrixError> {
        let mut at =  a.transpose();
        let mut adjoint_operator = try!(dot(&mut at,a));
        let mut e = try!(eigenvalues(&mut adjoint_operator, Eig :: Eigenvalues, Triangular::Upper));
        let s = try!(a.matrix_map(&|x : &f64| x.sqrt(), &mut e));
        Ok(s)
}
