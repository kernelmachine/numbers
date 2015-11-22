use std::error;
use std::fmt;


#[derive(Debug)]
pub enum MatrixError{
    MismatchedDimensions,
    NonSquareMatrix,
    MalformedMatrix,
    SingularMatrix,
    LapackComputationError,
    LapackInputError,
    UnknownError
}

impl fmt::Display for MatrixError{

    fn fmt(&self, f : &mut fmt::Formatter ) -> fmt::Result{

        match *self{
            MatrixError::MismatchedDimensions=> write!(f, "Operation cannot be performed. Mismatched dimensions."),
            MatrixError::NonSquareMatrix=> write!(f, "Operation cannot be performed. Matrix is not square."),
            MatrixError::MalformedMatrix => write!(f, "Matrix is malformed."),
            MatrixError::SingularMatrix => write!(f, "Operation cannot be performed. Matrix has zero determinant."),
            MatrixError::LapackComputationError => write!(f, "Failure in the course of computation."),
            MatrixError::LapackInputError => write!(f, "Illegal argument detected."),
            MatrixError::UnknownError => write!(f,"Unknown error, please submit bug.")
        }
    }
}

impl error::Error for MatrixError{

    fn  description(&self) -> &str {
        match *self{

            MatrixError::MismatchedDimensions => "Operation cannot be performed. Mismatched dimensions.",
            MatrixError::NonSquareMatrix => "Operation cannot be performed. Matrix is not square.",
            MatrixError::MalformedMatrix => "Matrix is malformed.",
            MatrixError::SingularMatrix => "Operation cannot be performed. Matrix has zero determinant.",
            MatrixError::LapackComputationError => "Failure in the course of computation.",
            MatrixError::LapackInputError => "Illegal argument detected.",
            MatrixError::UnknownError => "Unknown error, please submit bug."
        }

    }

    fn cause(&self) -> Option<&error::Error> {
      match *self {
          MatrixError::MismatchedDimensions => None,
          MatrixError::NonSquareMatrix => None,
          MatrixError::MalformedMatrix => None,
          MatrixError::SingularMatrix => None,
          MatrixError::LapackComputationError => None,
          MatrixError::LapackInputError => None,
          MatrixError::UnknownError => None
      }
  }


}
