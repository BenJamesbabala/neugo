package neugo

import (
	"errors"
	"fmt"
	"math/rand"
)

var (
	// dimension size error
	ErrDimension = errors.New("invalid dimensions")
	// lambda value for exp distribution error
	ErrLambda = errors.New("invalid lambda")
	// matrix coordinate error
	ErrCoordinate = errors.New("invalid coordinate")
)

// A matrix is defined with number of rows, number
// of columns, and data that's stored in the matrix.
type Matrix struct {
	r, c int       // number of rows and columns
	data []float64 // stored matrix data
}

// Create a new empty matrix given numbers of rows and columns.
func NewZeros(numRows, numColumns int) (*Matrix, error) {
	if numRows < 1 || numColumns < 1 {
		return nil, ErrDimension
	}
	len := numRows * numColumns
	return &Matrix{
		r:    numRows,
		c:    numColumns,
		data: make([]float64, len),
	}, nil
}

// Create a new matrix with values initialized to 1.
func NewOnes(numRows, numColumns int) (*Matrix, error) {
	if numRows < 1 || numColumns < 1 {
		return nil, ErrDimension
	}
	return &Matrix{
		r: numRows,
		c: numColumns,
		data: func() []float64 {
			len := numRows * numColumns
			dat := make([]float64, len)
			for i, _ := range dat {
				dat[i] = 1.0
			}
			return dat
		}(),
	}, nil
}

// Create a new identity matrix given number of rows/columns,
// which are the same.
func NewId(size int) (*Matrix, error) {
	if size < 1 {
		return nil, ErrDimension
	}
	return &Matrix{
		r: size,
		c: size,
		data: func() []float64 {
			len := size * size
			dat := make([]float64, len)
			for i := 0; i < size; i++ {
				dat[i*(size+1)] = 1.0
			}
			return dat
		}(),
	}, nil
}

// Create a new exponentially distributed random matrix given
// number of rows, number of columns, and lambda.
func NewExp(numRows, numColumns int, lambda float64) (*Matrix, error) {
	if lambda <= 0.0 {
		return nil, ErrLambda
	}
	if numRows < 0 || numColumns < 0 {
		return nil, ErrDimension
	}
	return &Matrix{
		r: numRows,
		c: numColumns,
		data: func() []float64 {
			len := numRows * numColumns
			dat := make([]float64, len)
			for i, _ := range dat {
				dat[i] = rand.ExpFloat64() / lambda
			}
			return dat
		}(),
	}, nil
}

// Create a new normally distributed random matrix given
// mean, standard deviation, number of rows, and number of columns.
func NewNorm(numRows, numColumns int, m, s float64) (*Matrix, error) {
	if numRows < 0 || numColumns < 0 {
		return nil, ErrDimension
	}
	return &Matrix{
		r: numRows,
		c: numColumns,
		data: func() []float64 {
			len := numRows * numColumns
			dat := make([]float64, len)
			for i, _ := range dat {
				dat[i] = rand.NormFloat64()*s + m
			}
			return dat
		}(),
	}, nil
}

// Get number of rows.
func (m *Matrix) NumRow() int {
	return m.r
}

// Get number of columns.
func (m *Matrix) NumColumn() int {
	return m.c
}

// Get matrix data.
func (m *Matrix) Data() []float64 {
	return m.data
}

// Get element at (x, y); return an error if out of range.
func (m *Matrix) Get(x, y int) (float64, error) {
	if !(0 <= x && x < m.r) || !(0 <= y && y < m.c) {
		return 0.0, ErrCoordinate
	}
	// otherwise, get data
	return m.data[m.r*x+y], nil
}

// Get row number x from the matrix in a float64 slice.
func (m *Matrix) GetRow(x int) ([]float64, error) {
	if !(0 <= x && x < m.r) {
		return nil, ErrCoordinate
	}
	row := make([]float64, m.c)
	for i, _ := range row {
		row[i] = m.data[m.r*x+i]
	}
	return row, nil
}

// Get column number x from the matrix in a float64 slice.
func (m *Matrix) GetCol(x int) ([]float64, error) {
	if !(0 <= x && x < m.r) {
		return nil, ErrCoordinate
	}
	col := make([]float64, m.r)
	for i, _ := range col {
		col[i] = m.data[x+i*m.r]
	}
	return col, nil
}

// Set element at (x, y) with a value; return an error if out of range.
func (m *Matrix) Set(v float64, x, y int) error {
	if !(0 <= x && x < m.r) || !(0 <= y && y < m.c) {
		return ErrCoordinate
	}
	m.data[m.r*x+y] = v
	return nil
}

// Copy other matrix.
func (m *Matrix) Copy(m1 *Matrix) {
	m.r = m1.r
	m.c = m1.c
	copy(m.data, m1.data)
}

// Scalar multiplication operation.
func (m *Matrix) Scalar(val float64) (*Matrix, error) {
	result, err := NewZeros(m.r, m.c)
	if err != nil {
		return nil, err
	}
	for i, d := range m.data {
		result.data[i] = d * val
	}
	return result, nil
}

// Function operation.
func (m *Matrix) Func(fn func(float64) float64) (*Matrix, error) {
	result, err := NewZeros(m.r, m.c)
	if err != nil {
		return nil, err
	}
	for i, d := range m.data {
		result.data[i] = fn(d)
	}
	return result, nil
}

// Matrix addition operation.
func (m *Matrix) Add(m1 *Matrix) (*Matrix, error) {
	if m.r != m1.r || m.c != m1.c {
		return nil, ErrDimension
	}
	result, err := NewZeros(m.r, m.c)
	if err != nil {
		return nil, err
	}
	for i, _ := range m1.data {
		result.data[i] = m.data[i] + m1.data[i]
	}
	return result, nil
}

// Matrix multiplication operation.
func (m *Matrix) Mult(m1 *Matrix) (*Matrix, error) {
	if m.c != m1.r {
		return nil, ErrDimension
	}
	result, err := NewZeros(m.r, m1.c)
	if err != nil {
		return nil, err
	}
	for i := 0; i < result.r; i++ {
		for j := 0; j < result.c; j++ {
			sum := 0.0
			for k := 0; k < m.c; k++ {
				a, err := m.Get(i, k)
				if err != nil {
					return nil, err
				}
				b, err := m1.Get(k, j)
				if err != nil {
					return nil, err
				}
				sum += (a * b)
			}
			result.Set(sum, i, j)
		}
	}
	return result, nil
}

// Multiplication with a vector (slice of float64). This function
// is meant to be used for neural network iteration.
func (m *Matrix) MultVec() []float64 {

}

// Print matrix in the rows and columns form.
func (m *Matrix) Print() {
	for i := 0; i < m.r; i++ {
		fmt.Printf("[ ")
		for j := 0; j < m.c; j++ {
			fmt.Printf("%8.3f ", m.data[m.r*i+j])
		}
		fmt.Printf("]\n")
	}
}
