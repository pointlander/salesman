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
	count := 0
	for i := 0; i < 1024; i++ {
		if test() {
			count++
		}
	}
	fmt.Println(count)
}

func test() bool {
	a := []float64{
		0, 20, 42, 35,
		20, 0, 30, 34,
		42, 30, 0, 12,
		35, 34, 12, 0,
	}
	if !*FlagDebug {
		for i := 0; i < Size; i++ {
			for j := i + 1; j < Size; j++ {
				value := float64(rand.Intn(8) + 1)
				a[i*4+j] = value
				a[j*4+i] = value
			}
		}
	}
	if *FlagDebug {
		for i := 0; i < Size; i++ {
			for j := 0; j < Size; j++ {
				fmt.Printf("%f ", a[i*4+j])
			}
			fmt.Printf("\n")
		}
	}
	var search func(sum float64, i int, nodes []int, visited [4]bool) (float64, []int)
	search = func(sum float64, i int, nodes []int, visited [4]bool) (float64, []int) {
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
	for i := 1; i < 4; i++ {
		s, n := search(0, i, []int{i}, [Size]bool{})
		if s < sum {
			sum, nodes = s, n
		}
	}
	if *FlagDebug {
		fmt.Println(sum, nodes)
	}

	graph := pagerank.NewGraph64()
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
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

	adjacency := mat.NewDense(Size, Size, a)
	var eig mat.Eigen
	ok := eig.Factorize(adjacency, mat.EigenRight)
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
		visited := [4]bool{}
		state := offset
		visited[state] = true
		total, loop := 0.0, make([]int, 0, 8)
		loop = append(loop, state)
		for i := 0; i < Size; i++ {
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

	ranks := mat.NewDense(Size, Size, nil)
	for i := 0; i < Size; i++ {
		for j := 0; j < Size; j++ {
			ranks.Set(i, j, real(vectors.At(i, j)))
		}
	}
	if *FlagDebug {
		Reduction("results", ranks)
	}

	return sum == minTotal
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
