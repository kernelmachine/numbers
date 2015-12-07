use super::{Matrix,Triangular, Trans, RectMat, SqMat, NonSingularMat};
use blas::*;
use lapack::dgetri;
use num::traits::Num;
use rand :: Rand;
use matrixerror::MatrixError;
use factorizations::*;



#[macro_export]
macro_rules! matrix_equal {
    ( $x:expr , $y : expr) => (
        {
            assert!($x.row_size == $y.row_size);
            assert!($x.col_size == $y.col_size);
            assert!($x.transpose  == $y.transpose);
            if let Ok(s) = &$x - &$y{
                for elem in s.elements{
                    assert!(elem.abs() < 1e-6)
                };
                return Ok(true)
            }
            Ok(false)
        }

    )
    }



/// Compute the dot product between two matrices.
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
///     let mut a = RectMat ::new(vec![1.0,2.0],2,1).ok().unwrap();
///     let mut b = RectMat ::new(vec![1.0,2.0],1,2).ok().unwrap();
///     let mut c = numbers::operations::dot(&mut a,&mut b).ok().unwrap();
///     let mut d = Matrix::new(vec![1.0,2.0,2.0,4.0],2,2).ok().unwrap();
///     matrix_equal!(c,d)
///}
/// ```

pub fn dot <T: RectMat> (a : &mut T, b : &mut T) -> Result<Matrix, MatrixError>{

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
        Ok(RectMat {
            elements : c,
            row_size : m,
            col_size : n,
            transpose : Trans :: Regular,
        })
}


/// Map a function to all elements of a matrix.
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
/// let mut a : Matrix<f64>= Matrix ::random(10,10);
/// let v = numbers::operations::matrix_map(&|&x| x+x, &mut a).ok().unwrap();
/// let e = numbers::operations::matrix_map(&|&x| x*2.0, &mut a).ok().unwrap();
/// matrix_equal!(e,v)
///}
/// ```

/// Get the upper triangular matrix.
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
/// let mut a : Matrix<f64>= Matrix ::new(vec![1.0,2.0,3.0,4.0],2,2).ok().unwrap();
/// let t  = numbers::operations::triu(&mut a,0).ok().unwrap();
/// let ans = Matrix :: new(vec![1.0,0.0,3.0,4.0],2,2).ok().unwrap();
/// matrix_equal!(t,ans)
///}
/// ```


/// Get the lower triangular matrix
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
/// let mut a : Matrix<f64>= Matrix ::new(vec![1.0,2.0,3.0,4.0],2,2).ok().unwrap();
/// let t  = numbers::operations::tril(&mut a,0).ok().unwrap();
/// let ans = Matrix :: new(vec![1.0,2.0,0.0,4.0],2,2).ok().unwrap();
/// matrix_equal!(t,ans)
///}
/// ```

/// Invert a square, non-singular matrix via LU factorization.
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
/// let mut b = Matrix :: new(vec![4.0,7.0,2.0,6.0], 2,2).ok().unwrap();
/// let mut inv = numbers::operations::inverse(&mut b).ok().unwrap();
/// let ans = Matrix :: new(vec![0.6,-0.7,-0.2,0.4], 2,2).ok().unwrap();
/// matrix_equal!(inv,ans)
///  }
/// ```

pub fn inverse<T: NonSingularMat>(a : &mut T ) ->Result<T,MatrixError> {

    if let Ok((l, ipiv)) = lufact (a){
        let n = l.col_size;
        let lda = l.row_size;
        let mut work = vec![0.0; 4 * n];
        let iwork = n as isize * 4;
        let mut info = 0;
        dgetri(n, &mut l.elements, lda, &ipiv,&mut work, iwork, &mut info);
        match info {
            x if x > 0 => return Err(MatrixError::LapackComputationError),
            0 => { let m = Ok(l.to_owned()); return m}
            x if x < 0 => return  Err(MatrixError::LapackInputError),
            _ => return Err(MatrixError::UnknownError),
        };

    }
    Err(MatrixError::SingularMatrix)
}


/// Get the trace of a matrix (the sum of its diagonal elements)
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
///     let mut a = Matrix :: new(vec![4.0,7.0,2.0,6.0, 5.0,7.0,2.0,3.0,3.0], 3,3).ok().unwrap();
///     let tr = numbers::operations::trace(&mut a);
///     assert!(tr-12.0 < 1e-14)
///  }
/// ```
pub fn trace<T: SqMat>(a: &mut T ) -> f64{
    let diag : Vec<f64> = a.diagonal();
    diag.iter().fold(0.0,|a,&b| a + b)

}

/// Get the product of the diagonal elements of a matrix.
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
///     let mut a = Matrix :: new(vec![4.0,7.0,2.0,6.0, 5.0,7.0,2.0,3.0,3.0], 3,3).ok().unwrap();
///     let tr = numbers::operations::prod_diag(&mut a);
///     assert!(tr-60.0 < 1e-14)
///  }
/// ```
pub fn prod_diag<T: SqMat> (a: &mut T ) -> f64{
    let diag : Vec<f64> = a.diagonal();
    diag.iter().fold(0.0,|a,&b| a * b)

}

/// Derive the pseudoinverse of a matrix via SVD.
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
///     let mut b = Matrix :: new(vec![4.0,7.0,5.0,10.0], 2,2).ok().unwrap();
///      let inv = numbers::operations::pseudoinverse(&mut b).ok().unwrap();
///     let pinv = numbers::operations::pseudoinverse(&mut b).ok().unwrap();
///     matrix_equal!(inv,pinv)
///  }
/// ```
pub fn pseudoinverse <T: NonSingularMat>(a : &mut T ) ->Result<T,MatrixError> {
    let (mut u,mut e, mut vt) = try!(svd(a));
    let inv_e = try!(a.matrix_map(&|x : &f64| if x > &0.0 { x.recip()} else { 0.0 },&mut e));
    let mut d = try!(dot(&mut u, &mut inv_e.transpose()));
    dot (&mut d, &mut vt)
}

/// Check whether a matrix is orthogonal, ie that A^T A = I.
  ///
  /// # Arguments
  ///
  /// * `Matrix` - Matrix of type f64
  ///
  /// # Example
  /// ```
  ///#[macro_use] extern crate numbers;
  /// use numbers::Matrix;
  /// pub fn main(){
  /// let mut a = Matrix :: new(vec![1.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0,1.0],3,3).ok().unwrap();
  /// assert!(numbers::operations::is_orthogonal(&mut a).ok().unwrap())
  ///}
  /// ```
pub fn is_orthogonal <T : SqMat> (a : &mut T ) -> Result<bool, MatrixError>{
    let mut at = a.transpose();
    let d = try!(dot (a, &mut at));
    let l : Matrix <f64> = Matrix::identity(a.row_size);
    matrix_equal!(d,l)
}

/// Check whether a matrix is normal, ie that A^T A = A A^T.
///
/// # Arguments
///
/// * `Matrix` - Matrix of type f64
///
/// # Example
/// ```
///#[macro_use] extern crate numbers;
/// use numbers::Matrix;
/// pub fn main(){
/// let mut a = Matrix :: new(vec![1.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0,1.0],3,3).ok().unwrap();
/// assert!(numbers::operations::is_normal(&mut a).ok().unwrap())
///}
/// ```
pub fn is_normal<T: SqMat>(a : &mut T) -> Result<bool, MatrixError> {

    let mut at = a.transpose();

    let inner = try!(dot (a, &mut at));
    let outer = try!(dot (&mut at, a));

    matrix_equal!(inner,outer)

}




/// Check whether a matrix is symmetric, ie that A^T = A.
///
/// # Arguments
///
/// * `Matrix` - Matrix of type f64
///
/// # Example
/// ```
///#[macro_use] extern crate numbers;
/// use numbers::Matrix;
/// pub fn main(){
/// let mut a = Matrix :: new(vec![4.0,5.0,5.0,4.0],2,2).ok().unwrap();
/// assert!(numbers::operations::is_symmetric(&mut a).ok().unwrap())
///}
/// ```
pub fn is_symmetric<T: SqMat> (a : &mut T) -> Result<bool, MatrixError>{

    let mut at = a.transpose();
    let a_row = a.row_size;
    let at_row = at.row_size;
    let at_dot = try!(dot(&mut at, &mut Matrix::identity(at_row)));
    let a_dot = try!(dot(a, &mut Matrix::identity(a_row)));
    matrix_equal!(at_dot,a_dot)
}
