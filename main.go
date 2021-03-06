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
	"github.com/pointlander/salesman/clusters"
	"github.com/pointlander/salesman/kmeans"
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

// Eigen2 uses eigen vectors to solve the traveling salesman problem
func Eigen2(a []float64) (float64, []int) {
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

	type Node struct {
		ID   int
		Rank float64
	}
	nodes := make([]Node, 0, 8)
	for i := 0; i < Size; i++ {
		for j := 0; j < Size; j++ {
			nodes = append(nodes, Node{
				ID:   i,
				Rank: math.Abs(real(vectors.At(i, j))),
			})
			nodes = append(nodes, Node{
				ID:   i,
				Rank: math.Abs(real(leftVectors.At(i, j))),
			})
		}
	}
	sort.Slice(nodes, func(i, j int) bool {
		return nodes[i].Rank < nodes[j].Rank
	})
	if *FlagDebug {
		for _, node := range nodes {
			fmt.Println(node)
		}
	}

	total, loop := math.MaxFloat64, make([]int, 0, 8)
	for i := 0; i < len(nodes); i++ {
		visited, l := make(map[int]bool), make([]int, 0, 8)
		for _, node := range nodes[i%len(nodes):] {
			if len(visited) == Size {
				break
			}
			if visited[node.ID] {
				continue
			}
			l = append(l, node.ID)
			visited[node.ID] = true
		}
		if len(visited) < Size {
			break
		}
		l = append(l, l[0])
		last, t := l[0], 0.0
		for _, node := range l[1:] {
			t += a[last*Size+node]
			last = node
		}
		if t < total {
			total, loop = t, l
		}
	}

	return total, loop
}

// Coordinates is a slice of float64
type Coordinates struct {
	ID     int
	Values []float64
}

// Coordinates implements the Observation interface for a plain set of float64
// coordinates
func (c Coordinates) Coordinates() clusters.Coordinates {
	return clusters.Coordinates(c.Values)
}

// Distance returns the euclidean distance between two coordinates
func (c Coordinates) Distance(p2 clusters.Coordinates) float64 {
	var r float64
	for i, v := range c.Values {
		r += math.Pow(v-p2[i], 2)
	}
	return r
}

// EigenKMeans uses eigen vectors and kmeans to solve the traveling salesman problem
func EigenKMeans(a []float64) (float64, []int) {
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

	min, max := math.MaxFloat64, -math.MaxFloat64
	for r := 0; r < Size; r++ {
		for c := 0; c < Size; c++ {
			value := real(values[c] * vectors.At(r, c))
			if value > max {
				max = value
			}
			if value < min {
				min = value
			}
		}
	}
	/*for r := 0; r < Size; r++ {
		for c := 0; c < Size; c++ {
			value := real(values[c] * leftVectors.At(r, c))
			if value > max {
				max = value
			}
			if value < min {
				min = value
			}
		}
	}*/
	var d clusters.Observations
	scale := max - min
	for r := 0; r < Size; r++ {
		row := Coordinates{
			ID: r,
		}
		for c := 0; c < Size; c++ {
			row.Values = append(row.Values, (real(values[c]*vectors.At(r, c))-min)/scale)
		}
		d = append(d, row)
	}
	/*for r := 0; r < Size; r++ {
		row := Coordinates{
			ID: id,
		}
		for c := 0; c < Size; c++ {
			row.Values = append(row.Values, (real(values[c]*leftVectors.At(r, c))-min)/scale)
		}
		d = append(d, row)
	}*/

	km := kmeans.New()
	clusters, err := km.Partition(d, 2)
	if err != nil {
		panic(err)
	}
	if *FlagDebug {
		size := 0
		values := make([]float64, 0, 8)
		for _, c := range clusters {
			values = append(values, c.Center...)
			size++
			for _, observation := range c.Observations {
				size++
				values = append(values, observation.(Coordinates).Values...)
			}
			fmt.Printf("Centered at x: %v\n", c.Center)
			fmt.Printf("Matching data points: %+v\n\n", c.Observations)
		}
		ranks := mat.NewDense(size, Size, values)
		fmt.Println(ranks)
		Reduction("kmeans", ranks)
	}

	return 0, nil
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

// Neural2 uses a neural network to solve the traveling salesman problem
func Neural2(a []float64) (float64, []int) {
	data := tf64.NewSet()
	data.Add("nodes", Size, Size*Size)
	data.Add("distances", 1, Size*Size)

	inputs := tf64.NewSet()
	inputs.Add("inputs", Size, 1)
	in := inputs.Weights[0]
	in.X = in.X[:cap(in.X)]

	nodes, distances := data.Weights[0], data.Weights[1]
	for i := 0; i < Size; i++ {
		for j := 0; j < Size; j++ {
			inputs := make([]float64, 4)
			inputs[i] = 1
			inputs[j] = 1
			nodes.X = append(nodes.X, inputs...)
			distances.X = append(distances.X, a[i*Size+j])
		}
	}

	set := tf64.NewSet()
	set.Add("aw", Size, Size)
	set.Add("bw", Size, 1)
	set.Add("ab", Size)
	set.Add("bb", 1, 1)

	for _, w := range set.Weights[:2] {
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, rand.NormFloat64()*factor)
		}
	}
	for _, w := range set.Weights[2:] {
		w.X = w.X[:cap(w.X)]
	}

	deltas := make([][]float64, 0, 8)
	for _, p := range set.Weights {
		deltas = append(deltas, make([]float64, len(p.X)))
	}

	l1 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("aw"), data.Get("nodes")), set.Get("ab")))
	l2 := tf64.Add(tf64.Mul(set.Get("bw"), l1), set.Get("bb"))
	cost := tf64.Avg(tf64.Quadratic(l2, data.Get("distances")))

	alpha, eta, iterations := .3, .3, 1024
	points := make(plotter.XYs, 0, iterations)
	i := 0
	for i < iterations {
		total := 0.0
		data.Zero()
		set.Zero()

		total += tf64.Gradient(cost).X[0]
		sum := 0.0
		for _, p := range set.Weights {
			for _, d := range p.D {
				sum += d * d
			}
		}
		norm := math.Sqrt(sum)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}

		for j, w := range set.Weights {
			for k, d := range w.D {
				deltas[j][k] = alpha*deltas[j][k] - eta*d*scaling
				set.Weights[j].X[k] += deltas[j][k]
			}
		}

		points = append(points, plotter.XY{X: float64(i), Y: total})
		if *FlagDebug {
			fmt.Println(i, total)
		}
		if total < .0001 {
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

		err = p.Save(8*vg.Inch, 8*vg.Inch, "cost_neural.png")
		if err != nil {
			panic(err)
		}
	}

	l1 = tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("aw"), inputs.Get("inputs")), set.Get("ab")))
	l2 = tf64.Add(tf64.Mul(set.Get("bw"), l1), set.Get("bb"))

	if *FlagDebug {
		for i := 0; i < Size; i++ {
			for j := 0; j < Size; j++ {
				in.X[j] = 0
			}
			in.X[i] = 1
			l2(func(a *tf64.V) bool {
				fmt.Println(i, a.X[0])
				return true
			})
		}
	}

	aw := set.Weights[0]
	bw := set.Weights[1]
	ab := set.Weights[2]
	distance := make([]float64, Size*Size)
	for i := 0; i < Size; i++ {
		for j := 0; j < Size; j++ {
			if i == j {
				continue
			}
			sum := 0.0
			for k := 0; k < Size; k++ {
				x := (aw.X[k+i*Size]+ab.X[i])*bw.X[i] - (aw.X[k+j*Size]+ab.X[j])*bw.X[j]
				sum += x * x
			}
			distance[i*Size+j] = math.Sqrt(sum)
		}
	}
	if *FlagDebug {
		for i := 0; i < Size; i++ {
			for j := 0; j < Size; j++ {
				fmt.Printf("%f ", distance[i*Size+j])
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
				if v := distance[state*Size+j]; v < min {
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

	total0, loop0 := Search(a)
	total1, loop1 := PageRank(a)
	vectors, total2, loop2 := Eigen(a)
	total3, loop3 := Eigen2(a)
	total4, loop4 := NearestNeighbor(a)
	EigenKMeans(a)
	total5, loop5 := Neural2(a)

	ranks := mat.NewDense(Size, Size, nil)
	for i := 0; i < Size; i++ {
		for j := 0; j < Size; j++ {
			ranks.Set(i, j, real(vectors.At(i, j)))
		}
	}
	if *FlagDebug {
		fmt.Println("Search", total0, loop0)
		fmt.Println("PageRank", total1, loop1)
		fmt.Println("Eigen", total2, loop2)
		fmt.Println("Eigen2", total3, loop3)
		fmt.Println("NearestNeighbor", total4, loop4)
		fmt.Println("Neural2", total5, loop5)
		Reduction("results", ranks)
	}

	return total0 == total5, total0 == total4
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
	r, _ := ranks.Caps()
	fmt.Println(r)
	for i := 0; i < r; i++ {
		fmt.Println(proj.At(i, 0), proj.At(i, 1))
		points = append(points, plotter.XY{X: proj.At(i, 0), Y: proj.At(i, 1)})
	}

	for i := 0; i < r; i++ {
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
