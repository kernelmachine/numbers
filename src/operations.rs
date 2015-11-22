use super::{Matrix,Triangular, Eig};
use blas::*;
use lapack::dgetri;
use num::traits::Num;
use rand :: Rand;
use matrixerror::MatrixError;
use factorizations::*;
use eigenvalues::*;
/// Compute dot product between two matrices.
pub fn dot (a : &mut Matrix<f64>, b : &mut Matrix<f64>) -> Result<Matrix<f64>, MatrixError>{
        if a.col_size != b.row_size {
            return Err(MatrixError::MismatchedDimensions)
        }

        let m = a.row_size;
        let n = b.col_size;
        let k = a.col_size;


        let mut c = vec![0.0; m*n];
        dgemm(b'N', b'N', m, n, k, 1.0, &a.elements, m, &b.elements,k, 0.0,&mut c, m);
        Ok(Matrix {
            elements : c,
            row_size : m,
            col_size : n,
            transpose : false,
        })
}

/// Compute kronecker product between two matrices.
pub fn kronecker(){
    unimplemented!();
}

/// Compute cross product between two matrices.
pub fn cross() {
    unimplemented!()
}

/// Map a function to all elements of matrix.
pub fn matrix_map <T: Num + Clone + Rand> (func : &Fn(&T) -> T, a : &mut Matrix<T>) -> Result<Matrix<T>, MatrixError>{
      Ok(Matrix {
           elements: a.elements.iter().map(func).collect(),
           row_size : a.row_size,
           col_size : a.col_size,
           transpose : false,
       })
}

/// Get upper triangular matrix
pub fn triu<T : Num + Clone + Rand>(a: &mut Matrix<T>, k: usize ) -> Result<Matrix<T>, MatrixError>{
    let tri_mat = Matrix :: tri(a.row_size, a.col_size, k, Triangular::Upper);
    &tri_mat *&a
}

/// Get lower triangular matrix
pub fn tril<T : Num + Clone + Rand>(a: &mut Matrix<T>, k: usize ) -> Result<Matrix<T>, MatrixError>{
    let tri_mat = Matrix :: tri(a.row_size, a.col_size, k, Triangular::Lower);
    &tri_mat *&a
}

/// Invert a matrix via LU factorization.
pub fn inverse(a : &mut Matrix<f64> ) ->Result<Matrix<f64>,MatrixError> {

    if let Ok((l, ipiv)) = lu (a){
        let n = l.col_size;
        let lda = l.row_size;
        let mut work = vec![0.0; 4 * n];
        let iwork = n as isize * 4;
        let mut info = 0;
        dgetri(n, &mut l.elements, lda, &ipiv,&mut work, iwork, &mut info);
        match info {
            1 => return Err(MatrixError::LapackComputationError),
            0 => { let m = Ok(l.to_owned()); return m}
            -1 => return  Err(MatrixError::LapackInputError),
            _ => return Err(MatrixError::UnknownError),
        };

    }
    Err(MatrixError::SingularMatrix)
}

/// Derive the pseudoinverse of a matrix via SVD.
pub fn pseudoinverse(a : &mut Matrix<f64> ) ->Result<Matrix<f64>,MatrixError> {

    if let Ok((mut u,mut e, mut vt)) = svd(a) {
        let inv_e = matrix_map(&|x : &f64| if x > &0.0 { x.recip()} else { 0.0 },&mut e);
        let d = dot(&mut u, &mut inv_e.ok().unwrap().transpose());
        let m = dot (&mut d.ok().unwrap(), &mut vt);
        return m
    }
    Err(MatrixError::LapackComputationError)

}

/// Get the nullspace (kernel) of a matrix. (x where Ax = 0)
pub fn nullspace(){
    unimplemented!();
}

/// Check whether a matrix is unitary. A^T A = I
pub fn is_unitary(a : &mut Matrix<f64> ) -> Result<bool, MatrixError>{
    if a.row_size != a.col_size {
        return Err(MatrixError::NonSquareMatrix)
    }

    let mut at = a.transpose();
    let d = dot (a, &mut at).ok().unwrap();
    let l : Matrix <f64> = Matrix::identity(a.row_size);
    Ok(d == l)
}

/// Check whether a matrix is diagonalizable
pub fn is_diagonalizable(a : &mut Matrix <f64>) -> Result<bool, MatrixError> {
    if a.row_size != a.col_size {
        return Err(MatrixError::NonSquareMatrix)
    }

    if is_normal(a).ok().unwrap(){ println!("normal"); Ok(true)}
    else{
        Ok(eigenvalues(a, Eig::Eigenvalues, Triangular::Upper).ok().unwrap().elements.len() == a.row_size)
    }
}

/// Check whether a matrix is normal. A^T A = A A ^ T
pub fn is_normal(a : &mut Matrix<f64>) -> Result<bool, MatrixError> {
    if a.row_size != a.col_size {
        return Err(MatrixError::NonSquareMatrix)
    }
    let mut at = a.transpose();
    let inner = dot (a, &mut at).ok().unwrap();
    let outer = dot (&mut at, a).ok().unwrap();
    Ok(inner==outer)

}




/// Check whether a matrix is symmetric. A^T == A.
pub fn is_symmetric(a : &mut Matrix<f64>) -> Result<bool, MatrixError>{
    if a.row_size != a.col_size {
        return Err(MatrixError::NonSquareMatrix)
    }
    let mut at = a.transpose();
    Ok(a == &mut at)

}
