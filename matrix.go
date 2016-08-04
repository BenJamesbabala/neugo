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

// Transposition operation
func (m *Matrix) T() {

}

// Scalar multiplication operation.
func (m *Matrix) Scalar(val float64) {
	for i, _ := range m.data {
		m.data[i] *= val
	}
}

// Matrix addition operation.
func (m *Matrix) Add(m1 Matrix) error {
	if m.r != m1.r || m.c != m1.c {
		return ErrDimension
	}
	for i, d := range m1.data {
		m.data[i] += d
	}
	return nil
}

// Matrix multiplication operation.
func (m *Matrix) Mult(m1 Matrix) {

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
