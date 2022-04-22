// Copyright 2022 The Salesman Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"os"
	"sort"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/gradient/tf64"
	"github.com/pointlander/pagerank"
)

const (
	// Size is the size of the matrix
	Size = 4
)

var (
	// FlagDebug debug mode
	FlagDebug = flag.Bool("debug", false, "debug mode")
)

func main() {
	flag.Parse()
	rand.Seed(1)
	if *FlagDebug {
		test()
		return
	}
	eigenCount, nnCount := 0, 0
	for i := 0; i < 1024; i++ {
		eigen, nn := test()
		if eigen {
			eigenCount++
		}
		if nn {
			nnCount++
		}
	}
	fmt.Println(float64(eigenCount)/1024.0, float64(nnCount)/1024.0)
}

// Search searches for a solution to the traveling salesman problem
func Search(a []float64) (float64, []int) {
	var search func(sum float64, i int, nodes []int, visited [Size]bool) (float64, []int)
	search = func(sum float64, i int, nodes []int, visited [Size]bool) (float64, []int) {
		smallest, cities := math.MaxFloat64, nodes
		visited[i] = true
		skipped := true
		for j, skip := range visited {
			if skip {
				continue
			}
			skipped = false
			value, x := search(sum+a[i*Size+j], j, append(nodes, j), visited)
			if value < smallest {
				smallest, cities = value, x
			}
		}
		if skipped {
			return sum + a[i*Size+nodes[0]], append(cities, nodes[0])
		}
		return smallest, cities
	}
	sum, nodes := search(0, 0, []int{0}, [Size]bool{})
	for i := 1; i < Size; i++ {
		s, n := search(0, i, []int{i}, [Size]bool{})
		if s < sum {
			sum, nodes = s, n
		}
	}
	if *FlagDebug {
		fmt.Println(sum, nodes)
	}
	return sum, nodes
}

// PageRank uses page rank to solve the traveling salesman problem
func PageRank(a []float64) (float64, []uint64) {
	graph := pagerank.NewGraph64()
	for i := 0; i < Size; i++ {
		for j := 0; j < Size; j++ {
			if i == j {
				continue
			}
			graph.Link(uint64(i), uint64(j), a[i*Size+j])
		}
	}
	type City struct {
		ID   uint64
		Rank float64
	}
	cities := make([]City, 0, 8)
	graph.Rank(.85, 0.000001, func(node uint64, rank float64) {
		cities = append(cities, City{
			ID:   node,
			Rank: rank,
		})
	})
	sort.Slice(cities, func(i, j int) bool {
		return cities[i].Rank < cities[j].Rank
	})
	if *FlagDebug {
		fmt.Println(cities)
	}
	pageNodes := make([]uint64, 0, 8)
	pageNodes = append(pageNodes, cities[len(cities)-1].ID)
	for _, city := range cities {
		pageNodes = append(pageNodes, city.ID)
	}
	total := 0.0
	last := pageNodes[0]
	for _, node := range pageNodes[1:] {
		total += a[last*Size+node]
		last = node
	}
	if *FlagDebug {
		fmt.Println(total, pageNodes)
	}
	return total, pageNodes
}

// Eigen uses eigen vectors to solve the traveling salesman problem
func Eigen(a []float64) (*mat.CDense, float64, []int) {
	adjacency := mat.NewDense(Size, Size, a)
	var eig mat.Eigen
	ok := eig.Factorize(adjacency, mat.EigenBoth)
	if !ok {
		panic("Eigendecomposition failed")
	}

	values := eig.Values(nil)
	if *FlagDebug {
		for i, value := range values {
			fmt.Println(i, value, cmplx.Abs(value), cmplx.Phase(value))
		}
		fmt.Printf("\n")
	}

	vectors := mat.CDense{}
	eig.VectorsTo(&vectors)
	if *FlagDebug {
		for i := 0; i < Size; i++ {
			for j := 0; j < Size; j++ {
				fmt.Printf("%f ", vectors.At(i, j))
			}
			fmt.Printf("\n")
		}
		fmt.Printf("\n")
	}

	leftVectors := mat.CDense{}
	eig.LeftVectorsTo(&leftVectors)
	if *FlagDebug {
		for i := 0; i < Size; i++ {
			for j := 0; j < Size; j++ {
				fmt.Printf("%f ", leftVectors.At(i, j))
			}
			fmt.Printf("\n")
		}
		fmt.Printf("\n")
	}

	distances := make([]float64, Size*Size)
	for i := 0; i < Size; i++ {
		for j := 0; j < Size; j++ {
			if i == j {
				continue
			}
			sum := 0.0
			for k := 0; k < Size; k++ {
				x := real(values[k]*vectors.At(i, k)) - real(values[k]*vectors.At(j, k))
				sum += x * x
			}
			distances[i*Size+j] = math.Sqrt(sum) * a[i*Size+j]
		}
	}
	if *FlagDebug {
		for i := 0; i < Size; i++ {
			for j := 0; j < Size; j++ {
				fmt.Printf("%f ", distances[i*Size+j])
			}
			fmt.Printf("\n")
		}
	}

	leftDistances := make([]float64, Size*Size)
	for i := 0; i < Size; i++ {
		for j := 0; j < Size; j++ {
			if i == j {
				continue
			}
			sum := 0.0
			for k := 0; k < Size; k++ {
				x := real(values[k]*leftVectors.At(i, k)) - real(values[k]*leftVectors.At(j, k))
				sum += x * x
			}
			leftDistances[i*Size+j] = math.Sqrt(sum) * a[i*Size+j]
		}
	}
	if *FlagDebug {
		for i := 0; i < Size; i++ {
			for j := 0; j < Size; j++ {
				fmt.Printf("%f ", leftDistances[i*Size+j])
			}
			fmt.Printf("\n")
		}
	}

	minTotal, minLoop := math.MaxFloat64, make([]int, 0, 8)
	for offset := 0; offset < Size; offset++ {
		visited := [Size]bool{}
		state := offset
		visited[state] = true
		total, loop := 0.0, make([]int, 0, 8)
		loop = append(loop, state)
		for i := 0; i < Size-1; i++ {
			min, k := math.MaxFloat64, 0
			for j := 0; j < Size; j++ {
				if j == state || visited[j] {
					continue
				}
				if v := distances[state*Size+j]; v < min {
					min, k = v, j
				}
			}
			state = k
			visited[state] = true
			loop = append(loop, state)
		}
		loop = append(loop, loop[0])
		last := loop[0]
		for _, node := range loop[1:] {
			total += a[last*Size+node]
			last = node
		}
		if total < minTotal && loop[0] == loop[Size] {
			minTotal, minLoop = total, loop
		}
	}

	for offset := 0; offset < Size; offset++ {
		visited := [Size]bool{}
		state := offset
		visited[state] = true
		total, loop := 0.0, make([]int, 0, 8)
		loop = append(loop, state)
		for i := 0; i < Size-1; i++ {
			min, k := math.MaxFloat64, 0
			for j := 0; j < Size; j++ {
				if j == state || visited[j] {
					continue
				}
				if v := leftDistances[state*Size+j]; v < min {
					min, k = v, j
				}
			}
			state = k
			visited[state] = true
			loop = append(loop, state)
		}
		loop = append(loop, loop[0])
		last := loop[0]
		for _, node := range loop[1:] {
			total += a[last*Size+node]
			last = node
		}
		if total < minTotal && loop[0] == loop[Size] {
			minTotal, minLoop = total, loop
		}
	}
	if *FlagDebug {
		fmt.Println(minTotal, minLoop)
	}
	return &vectors, minTotal, minLoop
}

// NearestNeighbor uses nearest neighbor to solve the traveling salesman problem
func NearestNeighbor(a []float64) (float64, []int) {
	distances := a
	minTotal, minLoop := math.MaxFloat64, make([]int, 0, 8)
	for offset := 0; offset < Size; offset++ {
		visited := [Size]bool{}
		state := offset
		visited[state] = true
		total, loop := 0.0, make([]int, 0, 8)
		loop = append(loop, state)
		for i := 0; i < Size-1; i++ {
			min, k := math.MaxFloat64, 0
			for j := 0; j < Size; j++ {
				if j == state || visited[j] {
					continue
				}
				if v := distances[state*Size+j]; v < min {
					min, k = v, j
				}
			}
			state = k
			visited[state] = true
			loop = append(loop, state)
		}
		loop = append(loop, loop[0])
		last := loop[0]
		for _, node := range loop[1:] {
			total += a[last*Size+node]
			last = node
		}
		if total < minTotal && loop[0] == loop[Size] {
			minTotal, minLoop = total, loop
		}
	}
	return minTotal, minLoop
}

// Neural uses a neural network to solve the traveling salesman problem
func Neural(a []float64) (float64, []int) {
	Scale := 4
	set := tf64.NewSet()
	set.Add("A", Size, Size)
	set.Add("X", Size, Scale*Size)
	set.Add("B", Size)

	w := set.Weights[0]
	for i := 0; i < Size*Size; i++ {
		w.X = append(w.X, a[i])
	}

	w = set.Weights[1]
	factor := math.Sqrt(2.0 / float64(w.S[0]))
	for i := 0; i < cap(w.X); i++ {
		w.X = append(w.X, rand.NormFloat64()*factor)
	}

	set.Weights[2].X = set.Weights[2].X[:cap(set.Weights[2].X)]

	deltas := make([][]float64, 0, 8)
	for _, p := range set.Weights {
		deltas = append(deltas, make([]float64, len(p.X)))
	}

	l1 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("A"), set.Get("X")), set.Get("B")))
	cost := tf64.Avg(tf64.Quadratic(l1, set.Get("X")))

	alpha, eta, iterations := .3, .3, 1024
	points := make(plotter.XYs, 0, iterations)
	i := 0
	for i < iterations {
		total := 0.0
		set.Zero()

		total += tf64.Gradient(cost).X[0]
		sum := 0.0
		for _, p := range set.Weights[1:] {
			for _, d := range p.D {
				sum += d * d
			}
		}
		norm := math.Sqrt(sum)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}

		for j, w := range set.Weights[1:] {
			for k, d := range w.D {
				deltas[j+1][k] = alpha*deltas[j+1][k] - eta*d*scaling
				set.Weights[j+1].X[k] += deltas[j+1][k]
			}
		}

		points = append(points, plotter.XY{X: float64(i), Y: total})
		if *FlagDebug {
			fmt.Println(i, total)
		}
		if total < .01 {
			break
		}
		i++
	}

	if *FlagDebug {
		p := plot.New()

		p.Title.Text = "epochs vs cost"
		p.X.Label.Text = "epochs"
		p.Y.Label.Text = "cost"

		scatter, err := plotter.NewScatter(points)
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		p.Add(scatter)

		err = p.Save(8*vg.Inch, 8*vg.Inch, "cost.png")
		if err != nil {
			panic(err)
		}
	}

	distances := make([]float64, Size*Size)
	for i := 0; i < Size; i++ {
		for j := 0; j < Size; j++ {
			if i == j {
				continue
			}
			sum := 0.0
			for k := 0; k < Scale*Size; k++ {
				x := w.X[i+k*Size] - w.X[j+k*Size]
				sum += x * x
			}
			distances[i*Size+j] = math.Sqrt(sum)
		}
	}
	if *FlagDebug {
		for i := 0; i < Size; i++ {
			for j := 0; j < Size; j++ {
				fmt.Printf("%f ", distances[i*Size+j])
			}
			fmt.Printf("\n")
		}
	}
	minTotal, minLoop := math.MaxFloat64, make([]int, 0, 8)
	for offset := 0; offset < Size; offset++ {
		visited := [Size]bool{}
		state := offset
		visited[state] = true
		total, loop := 0.0, make([]int, 0, 8)
		loop = append(loop, state)
		for i := 0; i < Size; i++ {
			min, k := math.MaxFloat64, 0
			done := true
			for j := 0; j < Size; j++ {
				if j == state || visited[j] {
					continue
				}
				done = false
				if v := distances[state*Size+j]; v < min {
					min, k = v, j
				}
			}
			if done {
				loop = append(loop, loop[0])
				break
			}
			state = k
			visited[state] = true
			loop = append(loop, state)
		}
		last := loop[0]
		for _, node := range loop[1:] {
			total += a[last*Size+node]
			last = node
		}
		if total < minTotal && loop[0] == loop[Size] {
			minTotal, minLoop = total, loop
		}
	}
	if *FlagDebug {
		fmt.Println(minTotal, minLoop)
	}
	return minTotal, minLoop
}

func test() (bool, bool) {
	a := []float64{
		0, 20, 42, 35,
		20, 0, 30, 34,
		42, 30, 0, 12,
		35, 34, 12, 0,
	}
	if !*FlagDebug {
		a = make([]float64, Size*Size)
		for i := 0; i < Size; i++ {
			for j := i + 1; j < Size; j++ {
				value := float64(rand.Intn(8) + 1)
				a[i*Size+j] = value
				a[j*Size+i] = value
			}
		}
	}
	if *FlagDebug {
		for i := 0; i < Size; i++ {
			for j := 0; j < Size; j++ {
				fmt.Printf("%f ", a[i*Size+j])
			}
			fmt.Printf("\n")
		}
	}

	sum, nodes := Search(a)
	total, pageNodes := PageRank(a)
	vectors, eigenTotal, eigenLoop := Eigen(a)
	nnTotal, nnLoop := NearestNeighbor(a)
	//neuralTotal, neuralLoop := Neural(a)
	_, _, _, _, _ = nodes, total, pageNodes, eigenLoop, nnLoop

	ranks := mat.NewDense(Size, Size, nil)
	for i := 0; i < Size; i++ {
		for j := 0; j < Size; j++ {
			ranks.Set(i, j, real(vectors.At(i, j)))
		}
	}
	if *FlagDebug {
		Reduction("results", ranks)
	}

	/*if sum != minTotal {
		fmt.Println(sum, minTotal)
	}*/

	return sum == eigenTotal, sum == nnTotal
}

// Reduction reduces the matrix
func Reduction(name string, ranks *mat.Dense) {
	var pc stat.PC
	ok := pc.PrincipalComponents(ranks, nil)
	if !ok {
		panic("PrincipalComponents failed")
	}
	k := 2
	var proj mat.Dense
	var vec mat.Dense
	pc.VectorsTo(&vec)
	proj.Mul(ranks, vec.Slice(0, Size, 0, k))

	fmt.Printf("\n")
	points := make(plotter.XYs, 0, 8)
	for i := 0; i < Size; i++ {
		fmt.Println(proj.At(i, 0), proj.At(i, 1))
		points = append(points, plotter.XY{X: proj.At(i, 0), Y: proj.At(i, 1)})
	}

	for i := 0; i < Size; i++ {
		fmt.Printf("%d ", i)
		a0, b0 := proj.At(i, 0), proj.At(i, 1)
		for j := 0; j < Size; j++ {
			if i == j {
				fmt.Printf("(%d 0) ", j)
				continue
			}
			a1, b1 := proj.At(j, 0), proj.At(j, 1)
			a, b := a0-a1, b0-b1
			distance := math.Sqrt(a*a + b*b)
			fmt.Printf("(%d %f) ", j, distance)
		}
		fmt.Printf("\n")
	}

	p := plot.New()

	p.Title.Text = "x vs y"
	p.X.Label.Text = "x"
	p.Y.Label.Text = "y"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(3)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("%s.png", name))
	if err != nil {
		panic(err)
	}

	output, err := os.Create(fmt.Sprintf("%s.dat", name))
	if err != nil {
		panic(err)
	}
	defer output.Close()
	for _, point := range points {
		fmt.Fprintf(output, "%f %f\n", point.X, point.Y)
	}
}
