// Command ndbench-go runs the Go/Gonum CPU backend for the ndbench suite.
package main

import (
	"errors"
	"flag"
	"fmt"
	"math"
	"os"

	"gonum.org/v1/gonum/mat"
)

const (
	defaultVectorIterations = 1_000_000
	defaultPointCount       = 100_000
	defaultMatrixSize       = 128
	cosineDimension         = 1024
	defaultCosineIterations = 1_000
)

type operation string

const (
	vector2    operation = "vector2"
	vector3    operation = "vector3"
	affine2    operation = "affine2"
	affine3    operation = "affine3"
	matvec     operation = "matvec"
	matmul     operation = "matmul"
	cosine1024 operation = "cosine1024"
	eigh       operation = "eigh"
)

type config struct {
	operation  operation
	size       int
	iterations int
}

func parseOperation(value string) (operation, error) {
	switch value {
	case "vector2", "vec2":
		return vector2, nil
	case "vector3", "vec3":
		return vector3, nil
	case "affine2":
		return affine2, nil
	case "affine3":
		return affine3, nil
	case "matvec":
		return matvec, nil
	case "matmul":
		return matmul, nil
	case "cosine1024", "cosine":
		return cosine1024, nil
	case "eigh", "diagonalize":
		return eigh, nil
	default:
		return "", fmt.Errorf("unknown operation %q; expected vector2, vector3, affine2, affine3, matvec, matmul, cosine1024, or eigh", value)
	}
}

func usage() string {
	return `Usage:
  ndbench-go --op <operation> [options]

Operations:
  vector2, vector3    low-dimensional vector arithmetic repeated --iterations times
  affine2, affine3    homogeneous affine transform of --size points
  matvec              dense matrix-vector product of a --size square matrix
  matmul              dense square matrix product of order --size
  cosine1024          cosine similarity of two f64 vectors of length 1024
  eigh                full symmetric eigendecomposition of order --size

Options:
  --op, --operation <name>  operation to run
  --size <N>                point count or matrix order, depending on operation
  --iterations <N>          repeat the operation N times
  --help                    show this message

Examples:
  ndbench-go --op vector3 --iterations 1000000
  ndbench-go --op matmul --size 256
  ndbench-go --op eigh --size 128
  ndbench-go --op cosine1024 --iterations 1000
`
}

func parseArgs(args []string) (config, bool, error) {
	fs := flag.NewFlagSet("ndbench-go", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	fs.Usage = func() {
		fmt.Fprint(fs.Output(), usage())
	}
	opFlag := fs.String("op", "", "operation to run")
	operationFlag := fs.String("operation", "", "operation to run (alias for --op)")
	sizeFlag := fs.Int("size", 0, "point count or matrix order")
	iterationsFlag := fs.Int("iterations", 0, "number of repetitions")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return config{}, true, nil
		}
		return config{}, false, err
	}
	if fs.NArg() != 0 {
		return config{}, false, fmt.Errorf("unexpected positional arguments: %v", fs.Args())
	}
	if *opFlag != "" && *operationFlag != "" {
		return config{}, false, errors.New("use only one of --op and --operation")
	}
	operationValue := *opFlag
	if operationValue == "" {
		operationValue = *operationFlag
	}
	if operationValue == "" {
		return config{}, false, errors.New("missing --op or --operation")
	}
	op, err := parseOperation(operationValue)
	if err != nil {
		return config{}, false, err
	}

	sizeSet := flagWasSet(fs, "size")
	iterationsSet := flagWasSet(fs, "iterations")
	size := *sizeFlag
	iterations := *iterationsFlag
	if !sizeSet {
		size = defaultSize(op)
	}
	if !iterationsSet {
		iterations = defaultIterations(op)
	}
	if size <= 0 {
		return config{}, false, errors.New("--size must be greater than zero")
	}
	if iterations <= 0 {
		return config{}, false, errors.New("--iterations must be greater than zero")
	}

	return config{operation: op, size: size, iterations: iterations}, false, nil
}

func flagWasSet(fs *flag.FlagSet, name string) bool {
	set := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == name {
			set = true
		}
	})
	return set
}

func defaultSize(op operation) int {
	switch op {
	case vector2, vector3:
		return 1
	case affine2, affine3:
		return defaultPointCount
	case matvec, matmul, eigh:
		return defaultMatrixSize
	case cosine1024:
		return cosineDimension
	default:
		panic("unreachable operation")
	}
}

func defaultIterations(op operation) int {
	switch op {
	case vector2, vector3:
		return defaultVectorIterations
	case cosine1024:
		return defaultCosineIterations
	default:
		return 1
	}
}

func main() {
	config, help, err := parseArgs(os.Args[1:])
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n\n%s", err, usage())
		os.Exit(1)
	}
	if help {
		fmt.Print(usage())
		return
	}
	checksum, err := run(config)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("checksum=%.17e\n", checksum)
}

func run(config config) (float64, error) {
	switch config.operation {
	case vector2:
		return runVector2(config.iterations), nil
	case vector3:
		return runVector3(config.iterations), nil
	case affine2:
		return runAffine(config.size, config.iterations, 2), nil
	case affine3:
		return runAffine(config.size, config.iterations, 3), nil
	case matvec:
		return runMatVec(config.size, config.iterations), nil
	case matmul:
		return runMatMul(config.size, config.iterations), nil
	case cosine1024:
		return runCosine1024(config.iterations), nil
	case eigh:
		return runEigh(config.size, config.iterations)
	default:
		return 0, fmt.Errorf("unsupported operation %q", config.operation)
	}
}

func scalar(i, j, salt int) float64 {
	value := (i*37 + j*17 + salt*13) % 101
	return 0.125 + float64(value)/101.0
}

func tinyTerm(iteration int) float64 {
	return float64(iteration&7) * 1.0e-12
}

func embeddingScalar(index, lane int) float64 {
	return scalar(index, lane, cosineDimension+lane)*2.0 - 1.0
}

func runVector2(iterations int) float64 {
	a := [2]float64{1.25, -2.5}
	b := [2]float64{-0.75, 3.0}
	var checksum float64
	for iteration := 0; iteration < iterations; iteration++ {
		summed := [2]float64{a[0] + b[0], a[1] + b[1]}
		dot := a[0]*b[0] + a[1]*b[1]
		norm := math.Sqrt(summed[0]*summed[0] + summed[1]*summed[1])
		checksum += summed[iteration&1] + dot + norm + tinyTerm(iteration)
	}
	return checksum
}

func runVector3(iterations int) float64 {
	a := [3]float64{1.25, -2.5, 0.75}
	b := [3]float64{-0.75, 3.0, 1.5}
	var checksum float64
	for iteration := 0; iteration < iterations; iteration++ {
		summed := [3]float64{a[0] + b[0], a[1] + b[1], a[2] + b[2]}
		dot := a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
		norm := math.Sqrt(summed[0]*summed[0] + summed[1]*summed[1] + summed[2]*summed[2])
		cross := [3]float64{
			a[1]*b[2] - a[2]*b[1],
			a[2]*b[0] - a[0]*b[2],
			a[0]*b[1] - a[1]*b[0],
		}
		checksum += summed[iteration%3] + dot + norm + cross[(iteration+1)%3] + tinyTerm(iteration)
	}
	return checksum
}

func affineValues(dimension int) []float64 {
	switch dimension {
	case 2:
		return []float64{1.1, -0.2, 0.5, 0.3, 0.9, -0.7, 0.0, 0.0, 1.0}
	case 3:
		return []float64{
			1.1, -0.2, 0.1, 0.5,
			0.3, 0.9, -0.15, -0.7,
			0.05, 0.2, 1.05, 0.4,
			0.0, 0.0, 0.0, 1.0,
		}
	default:
		panic("unsupported affine dimension")
	}
}

func affineInput(points, dimension int) []float64 {
	homogeneous := dimension + 1
	input := make([]float64, homogeneous*points)
	for row := 0; row < homogeneous; row++ {
		for column := 0; column < points; column++ {
			if row == dimension {
				input[row*points+column] = 1.0
			} else {
				input[row*points+column] = scalar(column, row, dimension)
			}
		}
	}
	return input
}

func runAffine(points, iterations, dimension int) float64 {
	homogeneous := dimension + 1
	transform := mat.NewDense(homogeneous, homogeneous, affineValues(dimension))
	input := mat.NewDense(homogeneous, points, affineInput(points, dimension))
	var checksum float64
	for iteration := 0; iteration < iterations; iteration++ {
		var output mat.Dense
		output.Mul(transform, input)
		data := output.RawMatrix().Data
		checksum += sum(data) + data[0] + data[dimension*points+points-1]
	}
	return checksum
}

func scalarMatrix(size, salt int) []float64 {
	matrix := make([]float64, size*size)
	for row := 0; row < size; row++ {
		for column := 0; column < size; column++ {
			matrix[row*size+column] = scalar(row, column, salt)
		}
	}
	return matrix
}

func runMatVec(size, iterations int) float64 {
	matrix := mat.NewDense(size, size, scalarMatrix(size, size))
	vectorData := make([]float64, size)
	for row := range vectorData {
		vectorData[row] = scalar(row, 0, size+1)
	}
	vector := mat.NewVecDense(size, vectorData)
	var checksum float64
	for iteration := 0; iteration < iterations; iteration++ {
		var output mat.VecDense
		output.MulVec(matrix, vector)
		data := output.RawVector().Data
		checksum += sum(data) + data[0]
	}
	return checksum
}

func runMatMul(size, iterations int) float64 {
	left := mat.NewDense(size, size, scalarMatrix(size, size))
	right := mat.NewDense(size, size, scalarMatrix(size, size+1))
	var checksum float64
	for iteration := 0; iteration < iterations; iteration++ {
		var output mat.Dense
		output.Mul(left, right)
		data := output.RawMatrix().Data
		checksum += sum(data) + data[0]
	}
	return checksum
}

func runCosine1024(iterations int) float64 {
	a := make([]float64, cosineDimension)
	b := make([]float64, cosineDimension)
	for index := range a {
		a[index] = embeddingScalar(index, 0)
		b[index] = embeddingScalar(index, 1)
	}
	var checksum float64
	for iteration := 0; iteration < iterations; iteration++ {
		var dot, normASquared, normBSquared float64
		for index := 0; index < cosineDimension; index++ {
			dot += a[index] * b[index]
			normASquared += a[index] * a[index]
			normBSquared += b[index] * b[index]
		}
		similarity := dot / (math.Sqrt(normASquared) * math.Sqrt(normBSquared))
		checksum += similarity + tinyTerm(iteration)
	}
	return checksum
}

func symmetricMatrix(size int) *mat.SymDense {
	data := make([]float64, size*size)
	for row := 0; row < size; row++ {
		for column := 0; column < size; column++ {
			if row == column {
				data[row*size+column] = float64(size + 2)
				continue
			}
			distance := math.Abs(float64(row - column))
			minimum, maximum := row, column
			if minimum > maximum {
				minimum, maximum = maximum, minimum
			}
			data[row*size+column] = 0.01 * scalar(minimum, maximum, size) / (1.0 + distance)
		}
	}
	return mat.NewSymDense(size, data)
}

func runEigh(size, iterations int) (float64, error) {
	matrix := symmetricMatrix(size)
	var checksum float64
	for iteration := 0; iteration < iterations; iteration++ {
		var eigen mat.EigenSym
		if ok := eigen.Factorize(matrix, true); !ok {
			return 0, errors.New("Gonum EigenSym factorization failed")
		}
		values := eigen.Values(nil)
		var vectors mat.Dense
		eigen.VectorsTo(&vectors)
		vectorData := vectors.RawMatrix().Data
		checksum += sum(values) + sumSquares(vectorData)
	}
	return checksum, nil
}

func sum(values []float64) float64 {
	var result float64
	for _, value := range values {
		result += value
	}
	return result
}

func sumSquares(values []float64) float64 {
	var result float64
	for _, value := range values {
		result += value * value
	}
	return result
}
