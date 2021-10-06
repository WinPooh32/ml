package ml

import (
	"encoding/csv"
	"fmt"
	"io"
	"strconv"
)

type (
	DType = float32

	Class  int
	Column []DType
	Row    []DType
)

type Dataset struct {
	lables []string
	data   [][]Column
}

func (d Dataset) Class(lable string) (columns []Column, idx int, ok bool) {
	for i, v := range d.lables {
		if v == lable {
			return d.data[i], i, true
		}
	}
	return nil, -1, false
}

func (d Dataset) Distribution(lable string) (distrib DType) {
	var totalItems int
	var colSize int
	for i, v := range d.lables {
		class := d.data[i]
		size := len(class[0])
		if v == lable {
			colSize = size
		}
		totalItems += size
	}
	if totalItems == 0 {
		return 0
	}
	return DType(float64(colSize) / float64(totalItems))
}

func (d Dataset) Lables() []string {
	return d.lables
}

func MakeDatasetFromCSV(r *csv.Reader, lableColumn int, testPortion float32) (learn, test Dataset, err error) {
	learn = Dataset{
		lables: []string{},
		data:   [][]Column{},
	}

	test = Dataset{
		lables: []string{},
		data:   [][]Column{},
	}

	var lable string
	var fields []DType

	for line := 0; ; line++ {
		fields = fields[:0]

		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return learn, test, fmt.Errorf("read csv row: %w", err)
		}

		for i, field := range record {
			if i == lableColumn {
				lable = field
				continue
			}
			num, err := strconv.ParseFloat(field, 32)
			if err != nil {
				return learn, test, fmt.Errorf("parse field: %w", err)
			}
			fields = append(fields, DType(num))
		}

		var columns []Column
		var ok bool

		if columns, _, ok = learn.Class(lable); !ok {
			columns = make([]Column, len(fields))

			learn.lables = append(learn.lables, lable)
			learn.data = append(learn.data, columns)
		}

		if len(columns) != len(fields) {
			return learn, test, fmt.Errorf("unexpected fields count=%d line=%d", len(fields), line)
		}

		for i, col := range columns {
			columns[i] = append(col, fields[i])
		}
	}

	if testPortion <= 0 {
		return learn, test, nil
	}

	for classIDX, lable := range learn.lables {
		class := learn.data[classIDX]

		test.lables = append(test.lables, lable)
		test.data = append(test.data, make([]Column, len(class)))

		var cutcount = int(float32(len(class[0])) * testPortion)

		for i, column := range class {
			div := len(column) - cutcount

			test.data[classIDX][i] = append(test.data[classIDX][i], column[div:]...)
			class[i] = column[:div]
		}
	}

	return learn, test, nil
}
