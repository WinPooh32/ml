package bayes

import (
	"encoding/csv"
	"log"
	"os"
	"testing"

	"github.com/WinPooh32/ml"
)

func fitCSV(t *testing.T, name string, lablescol int) {
	f, err := os.Open(name)
	if err != nil {
		t.Fatalf("open dataset file: %s", err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.Comma = ','
	r.ReuseRecord = true
	r.FieldsPerRecord = 0

	data, test, err := ml.MakeDatasetFromCSV(r, lablescol, 0.3)
	if err != nil {
		t.Fatalf("make dataset from csv: %s", err)
	}

	nb := NewFromDataset(data)

	lables := nb.classes
	successes := make([]float32, len(lables))
	totals := make([]float32, len(lables))
	probs := make([]DType, len(lables))

	var sumTotals float32
	var sumSuccess float32

	item := []DType{}

	for i, v := range lables {
		class, _, _ := test.Class(v)
		totals[i] = float32(len(class[0]))

		for row := range class[0] {
			item = item[:0]
			for col := range class {
				item = append(item, class[col][row])
			}

			nb.PredictTo(probs, item)

			classIdx := ml.Argmax(probs)
			log.Println(v, "==", lables[classIdx], probs)

			if lables[classIdx] == v {
				successes[classIdx]++
				sumSuccess++
			}
		}
	}

	rates := map[string]float32{}

	for i, v := range successes {
		total := totals[i]
		sumTotals += total
		if total == 0 {
			continue
		}
		rates[lables[i]] = v / total
	}

	t.Logf("    classes lables=%+v", lables)
	t.Logf("    classes totals=%+v", totals)
	t.Logf("     success rates=%+v", rates)
	t.Logf("           success=%f", sumSuccess)
	t.Logf("             toatl=%f", sumTotals)
	t.Logf("totall success rate=%f", sumSuccess/sumTotals)

	for k, v := range rates {
		if v < 0.9 {
			t.Fatalf("low success prediction rate for class=%s with rate=%f", k, v)
		}
	}
}

func TestNaiveBayes_FitIris(t *testing.T) {
	fitCSV(t, "../assets/iris.csv", 4)
}
