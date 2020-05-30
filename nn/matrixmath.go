package nn

import "errors"

// These are errors
var (
	ErrEmptyMatrix          = errors.New("empty matrix")
	ErrZeroLengthColsOrRows = errors.New("cannot create matrix with zero length columns or rows")
	ErrNumColNotEqualRows   = errors.New("number of columns in first matrix and number of rows in second matrix are not equal, cannot multiply")
)

// MatSize takes matrix a, returns number of rows (m), columns (n) and error
func MatSize(a [][]float64) (int, int, error) {
	if a == nil {
		return 0, 0, ErrEmptyMatrix
	}

	return len(a), len(a[0]), nil
}

// MatMul takes two matrices, a and b, returns product of their multiplication and an error
func MatMul(a, b [][]float64) ([][]float64, error) {
	// Get sizes of matrix a and b
	n, m1, err := MatSize(a)
	if err != nil {
		return nil, err
	}
	m2, p, err := MatSize(b)
	if err != nil {
		return nil, err
	}
	// Number of columns in the first matrix must be equal number of rows in the second
	if m1 != m2 {
		return nil, ErrNumColNotEqualRows
	}

	c, err := MatCreate(n, p)
	if err != nil {
		return nil, err
	}

	for i := 0; i < n; i++ {
		for j := 0; j < p; j++ {
			for r := 0; r < m1; r++ {
				c[i][j] += a[i][r] * b[r][j]
			}
		}
	}

	return c, nil
}

// MatScalMul takes matrix and a scalar float64, returns product of their multiplication and an error
func MatScalMul(a [][]float64, c float64) ([][]float64, error) {
	m, n, err := MatSize(a)
	if err != nil {
		return nil, err
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			a[i][j] = c * a[i][j]
		}
	}
	return a, nil
}

// MatCreate takes number of rows and columns, returns zero valued matrix and an error
func MatCreate(m, n int) ([][]float64, error) {
	if m == 0 || n == 0 {
		return nil, ErrZeroLengthColsOrRows
	}
	a := make([][]float64, m)
	for i := range a {
		a[i] = make([]float64, n)
	}

	return a, nil
}
