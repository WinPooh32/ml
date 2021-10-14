// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package dbscan

import "github.com/WinPooh32/ml"

type DType = ml.DType

// Point represents a cluster point which can measure distance to another point.
type Point interface {
	Name() string
	DistanceTo(Point) DType
}

// Cluster clusters the points by using DBSCAN method. It requires two parameters: epsilon and the
// minimum number of points required to form a dense region (minDensity). It starts with an
// arbitrary starting point that has not been visited. This point's ε-neighborhood is retrieved, and
// if it contains sufficiently many points, a cluster is started. Otherwise, the point is labeled as
// noise. Note that this point might later be found in a sufficiently sized ε-environment of a
// different point and hence be made part of a cluster.
func Cluster(minDensity int, epsilon DType, points ...Point) (clusters [][]Point) {
	visited := make(map[string]bool, len(points))
	for _, point := range points {
		neighbours := findNeighbours(point, points, epsilon)
		if len(neighbours)+1 >= minDensity {
			visited[point.Name()] = true
			cluster := []Point{point}
			cluster = expandCluster(cluster, neighbours, visited, minDensity, epsilon)

			if len(cluster) >= minDensity {
				clusters = append(clusters, cluster)
			}
		} else {
			visited[point.Name()] = false
		}
	}
	return clusters
}

// Finds the neighbours from given array, depends on epsolon , which determines
// the distance limit from the point
func findNeighbours(point Point, points []Point, epsilon DType) []Point {
	neighbours := make([]Point, 0)
	for _, potNeigb := range points {
		if point.Name() != potNeigb.Name() && potNeigb.DistanceTo(point) <= epsilon {
			neighbours = append(neighbours, potNeigb)
		}
	}
	return neighbours
}

// Try to expand existing clutser
func expandCluster(cluster, neighbours []Point, visited map[string]bool, minDensity int, eps DType) []Point {
	seed := make([]Point, len(neighbours))
	copy(seed, neighbours)

	// Create a new set for merging
	set := make(map[string]Point, len(cluster)+len(neighbours))
	merge(set, cluster...)

	// Merge all of the points
	for _, point := range seed {
		clustered, isVisited := visited[point.Name()]
		if !isVisited {
			currentNeighbours := findNeighbours(point, seed, eps)
			if len(currentNeighbours)+1 >= minDensity {
				visited[point.Name()] = true
				merge(set, currentNeighbours...)
			}
		}

		if isVisited && !clustered {
			visited[point.Name()] = true
			merge(set, point)
		}
	}

	// Flatten and return the cluster
	merged := make([]Point, 0, len(set))
	for _, v := range set {
		merged = append(merged, v)
	}
	return merged
}

func merge(dst map[string]Point, src ...Point) {
	for _, v := range src {
		dst[v.Name()] = v
	}
}
