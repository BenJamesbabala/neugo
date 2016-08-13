package neugo

import (
	"fmt"
	"log"
	"testing"
)

func TestMatrix(t *testing.T) {
	// zeros
	m, err := NewZeros(4, 4)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Zeros\n")
	m.Print()

	// ones
	m1, err := NewOnes(4, 4)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Ones\n")
	m1.Print()

	// exp
	m2, err := NewExp(3, 3, 0.5)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("ExpDist\n")
	m2.Print()

	// Norm
	m3, err := NewNorm(3, 3, 0.0, 6.0)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("NormDist\n")
	m3.Print()

	// Addition
	m4, err := m2.Add(m3)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Sum of ExpDist and NormDist\n")
	m4.Print()

	// Multiplication
	m5, err := m2.Mult(m3)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Product of ExpDist and NormDist\n")
	m5.Print()
}
