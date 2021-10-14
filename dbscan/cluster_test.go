// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package dbscan

import (
	"fmt"
	"log"
	"testing"

	"github.com/WinPooh32/math"
	"github.com/stretchr/testify/assert"
)

type SimplePoint struct {
	position DType
}

func (s SimplePoint) DistanceTo(c Point) DType {
	distance := math.Abs(c.(SimplePoint).position - s.position)
	return distance
}

func (s SimplePoint) Name() string {
	return fmt.Sprint(s.position)
}

func TestPutAll(t *testing.T) {
	testMap := make(map[string]Point)
	clusterList := []Point{
		SimplePoint{10},
		SimplePoint{12},
	}
	merge(testMap, clusterList...)
	mapSize := len(testMap)
	if mapSize != 2 {
		t.Errorf("Map does not contain expected size 2 but was %d", mapSize)
	}
}

//Test find neighbour function
func TestFindNeighbours(t *testing.T) {
	log.Println("Executing TestFindNeighbours")
	clusterList := []Point{
		SimplePoint{0},
		SimplePoint{1},
		SimplePoint{-1},
		SimplePoint{1.5},
		SimplePoint{-0.5},
	}

	eps := DType(1.01)
	neighbours := findNeighbours(clusterList[0], clusterList, eps)

	assert.Equal(t, 3, len(neighbours))
}

func TestExpandCluster(t *testing.T) {
	log.Println("Executing TestExpandCluster")
	expected := 4
	clusterList := []Point{
		SimplePoint{0},
		SimplePoint{1},
		SimplePoint{2},
		SimplePoint{2.1},
		SimplePoint{5},
	}

	eps := DType(1.0)
	minPts := 3
	visitMap := make(map[string]bool)
	cluster := make([]Point, 0)
	cluster = expandCluster(cluster, clusterList, visitMap, minPts, eps)
	assert.Equal(t, expected, len(cluster))
}

func TestCluster(t *testing.T) {
	clusters := Cluster(2, 1.0,
		SimplePoint{1},
		SimplePoint{0.5},
		SimplePoint{0},
		SimplePoint{5},
		SimplePoint{4.5},
		SimplePoint{4})

	assert.Equal(t, 2, len(clusters))
	if 2 == len(clusters) {
		assert.Equal(t, 3, len(clusters[0]))
		assert.Equal(t, 3, len(clusters[1]))
	}
}

func TestClusterNoData(t *testing.T) {
	log.Println("Executing TestClusterNoData")

	clusters := Cluster(3, 1.0)
	assert.Equal(t, 0, len(clusters))
}
