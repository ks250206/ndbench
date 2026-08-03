package main

import (
	"math"
	"testing"
)

func TestChecksumsMatchRawReferenceAtSizeEight(t *testing.T) {
	tests := []struct {
		name      string
		config    config
		expected  float64
		tolerance float64
	}{
		{name: "vector2", config: config{operation: vector2, size: 1, iterations: 1}, expected: -7.23039321881345209, tolerance: 1e-14},
		{name: "vector3", config: config{operation: vector3, size: 1, iterations: 1}, expected: -6.89150471698584965, tolerance: 1e-14},
		{name: "affine2", config: config{operation: affine2, size: 8, iterations: 1}, expected: 19.6184405940594040, tolerance: 1e-12},
		{name: "affine3", config: config{operation: affine3, size: 8, iterations: 1}, expected: 28.6269801980198011, tolerance: 1e-12},
		{name: "matvec", config: config{operation: matvec, size: 8, iterations: 1}, expected: 30.3813964317223792, tolerance: 1e-12},
		{name: "matmul", config: config{operation: matmul, size: 8, iterations: 1}, expected: 191.503970199000094, tolerance: 1e-10},
		{name: "cosine1024", config: config{operation: cosine1024, size: cosineDimension, iterations: 1}, expected: -0.0687357264609637503, tolerance: 1e-14},
		{name: "eigh", config: config{operation: eigh, size: 8, iterations: 1}, expected: 88.0, tolerance: 1e-8},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := run(test.config)
			if err != nil {
				t.Fatalf("run failed: %v", err)
			}
			if math.Abs(got-test.expected) > test.tolerance {
				t.Fatalf("checksum mismatch: got %.17e, want %.17e (tol %.1e)", got, test.expected, test.tolerance)
			}
		})
	}
}

func TestCLIArgumentAliasesAndDefaults(t *testing.T) {
	parsed, help, err := parseArgs([]string{"--operation", "vec2", "--iterations", "1"})
	if err != nil || help {
		t.Fatalf("parse failed: config=%+v help=%v err=%v", parsed, help, err)
	}
	if parsed.operation != vector2 || parsed.size != 1 || parsed.iterations != 1 {
		t.Fatalf("unexpected config: %+v", parsed)
	}
}
