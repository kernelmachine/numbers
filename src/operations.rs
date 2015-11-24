use super::{Matrix,Triangular, Trans};
use blas::*;
use lapack::dgetri;
use num::traits::Num;
use rand :: Rand;
use matrixerror::MatrixError;
use factorizations::*;
/// Compute dot product between two matrices.
pub fn dot (a : &mut Matrix<f64>, b : &mut Matrix<f64>) -> Result<Matrix<f64>, MatrixError>{
        if a.col_size != b.row_size {
            return Err(MatrixError::MismatchedDimensions)
        }

        let m = a.row_size;
        let n = b.col_size;
        let k = a.col_size;

        let a_trans = match a.transpose {
            Trans :: Regular => b'N',
            Trans :: Transpose => b'T',
        };
        let b_trans = match b.transpose {
            Trans :: Regular => b'N',
            Trans :: Transpose => b'T',
        };

        let mut c = vec![0.0; m*n];
        dgemm(a_trans, b_trans, m, n, k, 1.0, &a.elements, m, &b.elements,k, 0.0,&mut c, m);
        Ok(Matrix {
            elements : c,
            row_size : m,
            col_size : n,
            transpose : Trans :: Regular,
        })
}


/// Map a function to all elements of matrix.
pub fn matrix_map <T: Num + Clone + Rand> (func : &Fn(&T) -> T, a : &mut Matrix<T>) -> Result<Matrix<T>, MatrixError>{
      Ok(Matrix {
           elements: a.elements.iter().map(func).collect(),
           row_size : a.row_size,
           col_size : a.col_size,
           transpose : Trans :: Regular,
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

    if let Ok((l, ipiv)) = lufact (a){
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


/// Get the trace of a matrix (the sum of its diagonal elements)
pub fn trace(a: &mut Matrix<f64> ) -> f64{
    let diag : Vec<f64> = a.diagonal();
    diag.iter().fold(0.0,|a,&b| a + b)

}

/// Get the product of the diagonal elements of a matrix.
pub fn prod_diag(a: &mut Matrix<f64> ) -> f64{
    let diag : Vec<f64> = a.diagonal();
    diag.iter().fold(0.0,|a,&b| a * b)

}

/// Derive the pseudoinverse of a matrix via SVD.
pub fn pseudoinverse(a : &mut Matrix<f64> ) ->Result<Matrix<f64>,MatrixError> {

    if let Ok((mut u,mut e, mut vt)) = svd(a) {
        let inv_e = try!(matrix_map(&|x : &f64| if x > &0.0 { x.recip()} else { 0.0 },&mut e));
        let mut d = try!(dot(&mut u, &mut inv_e.transpose()));
        let m = try!(dot (&mut d, &mut vt));
        return Ok(m)
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
    let d = try!(dot (a, &mut at));
    let l : Matrix <f64> = Matrix::identity(a.row_size);
    Ok(d == l)
}

/// Check whether a matrix is normal. A^T A = A A ^ T
pub fn is_normal(a : &mut Matrix<f64>) -> Result<bool, MatrixError> {
    if a.row_size != a.col_size {
        return Err(MatrixError::NonSquareMatrix)
    }
    let mut at = a.transpose();

    let inner = try!(dot (a, &mut at));
    let outer = try!(dot (&mut at, a));
    println!("{:?}", inner);
    println!("{:?}", outer);
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
