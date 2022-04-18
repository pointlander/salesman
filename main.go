// Copyright 2022 The Salesman Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"sort"
	"strconv"

	"github.com/pointlander/pagerank"
)

func main() {
	verify()
	a := []float64{
		0, 20, 42, 35,
		20, 0, 30, 34,
		42, 30, 0, 12,
		35, 34, 12, 0,
	}
	graph := pagerank.NewGraph64()
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			graph.Link(uint64(i), uint64(j), a[i*4+j])
		}
	}
	type City struct {
		ID   uint64
		Rank float64
	}
	cities := make([]City, 0, 8)
	graph.Rank(0.85, 0.000001, func(node uint64, rank float64) {
		cities = append(cities, City{
			ID:   node,
			Rank: rank,
		})
	})
	sort.Slice(cities, func(i, j int) bool {
		return cities[i].Rank < cities[j].Rank
	})
	fmt.Println(cities)
}

// https://kommradhomer.medium.com/my-lazy-take-on-travelling-salesman-problem-in-golang-f7b913878c5
func verify() {
	var cities = [4][4]int{
		{0, 20, 42, 35},
		{20, 0, 30, 34},
		{42, 30, 0, 12},
		{35, 34, 12, 0},
	}

	gg := permutation([]int{1, 2, 3})

	total := 0

	shortestDistance := -1
	shortestPaths := []string{}

	for _, elem := range gg {

		fmt.Println("route:", routesToStr(elem))

		lastCity := 0

		for _, city := range elem {

			total += cities[lastCity][city]

			lastCity = city
		}

		total += cities[lastCity][0]

		fmt.Println("total distance:", total)

		if shortestDistance == -1 || shortestDistance > total {

			shortestDistance = total
			shortestPaths = append(shortestPaths, routesToStr(elem))
		}

		total = 0

	}

	fmt.Println()
	fmt.Println()
	fmt.Println("shortestDistance:", shortestDistance)
	fmt.Println("shortestPaths:", shortestPaths)
}

func permutation(xs []int) (permuts [][]int) {
	var rc func([]int, int)
	rc = func(a []int, k int) {
		if k == len(a) {
			permuts = append(permuts, append([]int{}, a...))
		} else {
			for i := k; i < len(xs); i++ {
				a[k], a[i] = a[i], a[k]
				rc(a, k+1)
				a[k], a[i] = a[i], a[k]
			}
		}
	}
	rc(xs, 0)

	return permuts
}

func rangeSlice(start, stop int) []int {
	if start > stop {
		panic("Slice ends before it started")
	}
	xs := make([]int, stop-start)
	for i := 0; i < len(xs); i++ {
		xs[i] = i + 1 + start
	}
	return xs
}

func routesToStr(arr []int) string {
	result := "(0,"

	for _, o := range arr {
		result += strconv.Itoa(o)
		result += ","
	}

	result += "0)"

	return result
}
