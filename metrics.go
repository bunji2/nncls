package nncls

// MetricsData : メトリクスを保持するデータ型
type MetricsData struct {
	numClass int
	tp       []int
	fp       []int
	fn       []int
	tn       []int
	total    int
	totalTP  int
	totalFP  int
	totalFN  int
	totalTN  int
}

// NewMetricsData :
func NewMetricsData(numClass int) (r *MetricsData) {
	r = &MetricsData{
		numClass: numClass,
		tp:       make([]int, numClass),
		fp:       make([]int, numClass),
		fn:       make([]int, numClass),
		tn:       make([]int, numClass),
	}
	return
}

// Add : クラスごとの予測と回答を追加する
func (md *MetricsData) Add(classID, pred, answer) {
	if pred == 1 && answer == 1 {
		md.tp[classID] = md.tp[classID] + 1
	} else if pred == 1 && answer == 0 {
		md.fp[classID] = md.fp[classID] + 1
	} else if pred == 0 && answer == 1 {
		md.fn[classID] = md.fn[classID] + 1
	} else { // pred == 0 && answer == 0
		md.tn[classID] = md.tn[classID] + 1
	}
	md.total = 0
	md.totalTP = 0
	md.totalFP = 0
	md.totalFN = 0
	md.totalTN = 0
}

// Total : 合計値
func (md *MetricsData) Total() (r int) {
	if md.total > 0 {
		r = md.total
		return
	}
	for classID := 0; classID < numClass; classID++ {
		r += md.tp[classID] + md.fp[classID] + md.fn[classID] + md.tn[classID]
	}
	return
}

// TotalTP : TPの合計値
func (md *MetricsData) TotalTP() (r int) {
	if md.totalTP > 0 {
		r = md.totalTP
		return
	}
	for classID := 0; classID < numClass; classID++ {
		r += md.tp[classID]
	}
	return
}

// TotalFP : FPの合計値
func (md *MetricsData) TotalFP() (r int) {
	if md.totalFP > 0 {
		r = md.totalFP
		return
	}
	for classID := 0; classID < numClass; classID++ {
		r += md.fp[classID]
	}
	return
}

// TotalFN : FNの合計値
func (md *MetricsData) TotalFN() (r int) {
	if md.totalFN > 0 {
		r = md.totalFN
		return
	}
	for classID := 0; classID < numClass; classID++ {
		r += md.fn[classID]
	}
	return
}

// TotalTN : TNの合計値
func (md *MetricsData) TotalTN() (r int) {
	if md.totalTN > 0 {
		r = md.totalTN
		return
	}
	for classID := 0; classID < numClass; classID++ {
		r += md.tn[classID]
	}
	return
}

// Precision : クラスごとの適合率
func (md *MetricsData) Precision(classID int) (r float32) {
	r = float32(md.tp[classID]) / float32(md.tp[classID]+md.fp[classID])
	return
}

// Recall : クラスごとの再現率
func (md *MetricsData) Recall(classID int) (r float32) {
	r = float32(md.tp[classID]) / float32(md.tp[classID]+md.fn[classID])
	return
}

// Accuracy : クラスごとの正解率
func (md *MetricsData) Accuracy(classID int) (r float32) {
	r = float32(md.tp[classID]+md.tn[classID]) / float32(md.tp[classID]+md.fp[classID]+md.fn[classID]+md.tn[classID])
	return
}

// MicroMetrics : 全体の Micro なメトリクス
func (md *MetricsData) MicroMetrics() (microPrecision, microRecall, microFMeasure, overallAccuracy float32) {
	totalTP := md.TocalTP()
	totalFP := md.TocalFP()
	totalFN := md.TocalFN()
	totalTN := md.TocalTN()
	microPrecision = float32(totalTP) / float32(totalTP+totalFP)
	microRecall = float32(totalTP) / float32(totalTP+totalFN)
	microFMeasure = microPrecision * microRecall * 2.0 / (microPrecision + microRecall)
	overallAccuracy = float32(totalTP+totalTN) / float32(totalTP+totalFP+totalFN+totalTN)
	return
}

// MacroMetrics : 全体の Macro なメトリクス
func (md *MetricsData) MacroMetrics() (macroPrecision, macroRecall, macroFMeasure, averageAccuracy float32) {
	p := 0.0
	r := 0.0
	a := 0.0
	for i := 0; i < md.numClass; i++ {
		p += md.Precision(i)
		r += md.Recall(i)
		a += md.Accuracy(i)
	}
	macroPrecision = p / md.numClass
	macroRecall = r / md.numClass
	macroFMeasure = macroPrecision * macroRecall * 2.0 / (macroPrecision + macroRecall)
	averageAccuracy = a / md.numClass
	return
}
