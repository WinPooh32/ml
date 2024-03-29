package bayes

import (
	"github.com/WinPooh32/ml"
	"github.com/WinPooh32/series/math"
)

type (
	DType = ml.DType

	Class  = ml.Class
	Column = ml.Column
	Row    = ml.Row
)

type NaiveBayes struct {
	classes  []string
	dim      int
	prob     []DType
	mean     []Row
	variance []Row
}

func New(classes []string, prob []DType, featuresDim int) *NaiveBayes {
	return &NaiveBayes{
		classes:  classes,
		dim:      featuresDim,
		prob:     prob,
		mean:     makeRows(len(classes), featuresDim),
		variance: makeRows(len(classes), featuresDim),
	}
}

func NewFromDataset(data ml.Dataset) *NaiveBayes {
	lables := data.Lables()
	distrib := make([]DType, 0, len(lables))

	for _, v := range lables {
		distrib = append(distrib, data.Distribution(v))
	}

	class, _, _ := data.Class(lables[0])

	nb := New(lables, distrib, len(class))

	ds := make([][]Column, 0, len(lables))
	for _, v := range lables {
		class, _, _ := data.Class(v)
		ds = append(ds, class)
	}

	nb.Fit(ds)

	return nb
}

func (nb *NaiveBayes) SetProb(prob []DType) {
	nb.prob = prob
}

func (nb *NaiveBayes) Fit(dataset [][]Column) {
	if len(dataset) == 0 || len(dataset[0]) != int(nb.dim) {
		panic("mismatched diminitions!")
	}
	for class, object := range dataset {
		for feature, column := range object {
			mean := nb.calcMean(column)
			nb.mean[class][feature] = mean
			nb.variance[class][feature] = nb.calcVariance(column, mean)
		}
	}
}

func (nb *NaiveBayes) PredictTo(probs []DType, object []DType) {
	clCount := len(nb.classes)
	clProbs := nb.prob
	clMeans := nb.mean
	clVaris := nb.variance

	if len(object) != int(nb.dim) ||
		len(probs) != clCount ||
		len(clProbs) != clCount ||
		len(clMeans) != clCount ||
		len(clVaris) != clCount {
		panic("mismatched diminitions!")
	}

	for class := range nb.classes {
		proba := clProbs[class]
		means := clMeans[class]
		variances := clVaris[class]
		p := nb.calcPosterior(proba, object, means, variances)
		probs[class] = p
	}

	sum := nb.sumExp(probs)

	for i, v := range probs {
		v = math.Exp(v)
		q := math.Log(v / (sum - v))
		exp := math.Exp(q)

		probs[i] = exp / (1 + exp)
	}

	clCount = len(nb.classes)
	if len(probs) != clCount {
		panic("")
	}
}

func (nb *NaiveBayes) calcPosterior(
	proba DType,
	object []DType,
	means,
	variances Row,
) (p DType) {
	// Satisfy bounds checker.
	size := nb.dim
	if len(object) < size ||
		len(means) < size ||
		len(variances) < size {
		panic("bad size")
	}

	p = math.Log(proba)

	for feat := 0; feat < size; feat++ {
		value := object[feat]
		mean := means[feat]
		vari := variances[feat]
		if vari == 0 {
			continue
		}
		g := nb.calcGauss(value, mean, vari)
		p += math.Log1p(g)
	}

	return p
}

func (nb *NaiveBayes) calcGauss(val DType, mean DType, vari DType) DType {
	s := val - mean
	m := -(s * s) / (2.0 * vari)
	g := math.Exp(m)
	return g / math.Sqrt(2*math.Pi*vari)
}

func (nb *NaiveBayes) calcMean(col Column) DType {
	var mean DType = 0.0
	for _, val := range col {
		mean += val / DType(len(col))
	}
	return mean
}

func (nb *NaiveBayes) calcVariance(col Column, mean DType) DType {
	var c DType = 1.0 / DType(len(col))
	var vari DType = 0.0
	for _, val := range col {
		s := val - mean
		vari += s * s * c
	}
	return vari
}

func (nb *NaiveBayes) sumExp(values []DType) (s DType) {
	for _, v := range values {
		s += math.Exp(v)
	}
	return s
}

func makeRows(n int, dim int) []Row {
	m := make([]Row, n)
	for i := 0; i < n; i++ {
		m[Class(i)] = make(Row, dim)
	}
	return m
}
