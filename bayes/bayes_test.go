package bayes

import (
	"encoding/csv"
	"log"
	"os"
	"testing"

	"github.com/WinPooh32/ml"
)

func TestNaiveBayes_Fit(t *testing.T) {

	f, err := os.Open("../assets/iris.csv")
	if err != nil {
		t.Fatalf("open dataset file: %s", err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.Comma = ','
	r.ReuseRecord = true
	r.FieldsPerRecord = 5

	data, test, err := ml.MakeDatasetFromCSV(r, 4, 0.3)
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

	var item = []DType{}

	for i, v := range lables {
		class, _, _ := test.Class(v)
		totals[i] = float32(len(class[0]))

		for row := range class[0] {
			item = item[:0]
			for col := range class {
				item = append(item, class[col][row])
			}

			nb.PredictTo(probs, item)

			classIdx := ml.Argmin(probs)
			log.Println(v, "==", lables[classIdx])

			if lables[classIdx] == v {
				successes[classIdx]++
				sumSuccess++
			}
		}
	}

	var rates = map[string]float32{}

	for i, v := range successes {
		total := totals[i]
		sumTotals += total
		if total == 0 {
			continue
		}
		rates[lables[i]] = v / total
	}

	t.Logf("     classs lables=%+v", lables)
	t.Logf("     classs totals=%+v", totals)
	t.Logf("     success rates=%+v", rates)
	t.Logf("           success=%f", sumSuccess)
	t.Logf("             toatl=%f", sumTotals)
	t.Logf("toatl success rate=%f", sumSuccess/sumTotals)

	for k, v := range rates {
		if v < 0.9 {
			t.Fatalf("low success prediction rate for class=%s with rate=%f", k, v)
		}
	}
}
