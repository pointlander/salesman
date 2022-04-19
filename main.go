// Copyright 2022 The Salesman Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/pointlander/pagerank"
)

func main() {
	rand.Seed(1)
	a := []float64{
		0, 20, 42, 35,
		20, 0, 30, 34,
		42, 30, 0, 12,
		35, 34, 12, 0,
	}
	for i := 0; i < 4; i++ {
		for j := i + 1; j < 4; j++ {
			value := float64(rand.Intn(8) + 1)
			a[i*4+j] = value
			a[j*4+i] = value
		}
	}
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			fmt.Printf("%f ", a[i*4+j])
		}
		fmt.Printf("\n")
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
			value, x := search(sum+a[i*4+j], j, append(nodes, j), visited)
			if value < smallest {
				smallest, cities = value, x
			}
		}
		if skipped {
			return sum + a[i*4+nodes[0]], append(cities, nodes[0])
		}
		return smallest, cities
	}
	sum, nodes := search(0, 0, []int{0}, [4]bool{})
	for i := 1; i < 4; i++ {
		s, n := search(0, i, []int{i}, [4]bool{})
		if s < sum {
			sum, nodes = s, n
		}
	}
	fmt.Println(sum, nodes)

	graph := pagerank.NewGraph64()
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			if i == j {
				continue
			}
			graph.Link(uint64(i), uint64(j), a[i*4+j])
		}
	}
	type City struct {
		ID   uint64
		Rank float64
	}
	cities := make([]City, 0, 8)
	graph.Rank(1, 0.000001, func(node uint64, rank float64) {
		cities = append(cities, City{
			ID:   node,
			Rank: rank,
		})
	})
	sort.Slice(cities, func(i, j int) bool {
		return cities[i].Rank < cities[j].Rank
	})
	fmt.Println(cities)
	pageNodes := make([]uint64, 0, 8)
	pageNodes = append(pageNodes, cities[len(cities)-1].ID)
	for _, city := range cities {
		pageNodes = append(pageNodes, city.ID)
	}
	total := 0.0
	last := pageNodes[0]
	for _, node := range pageNodes[1:] {
		total += a[last*4+node]
		last = node
	}
	fmt.Println(total, pageNodes)
}
