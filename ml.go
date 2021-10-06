package ml

import "math"

func Argmin(series []DType) int {
	var min DType = math.MaxFloat32
	var offset int = -1
	for i, v := range series {
		if v < min {
			min = v
			offset = i
		}
	}
	return offset
}
