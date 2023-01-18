Package `interinhib` provides inter-layer inhibition params, which can be added to Layer types. 

Note: it is better to use direct inhibitory projections -- try that first before using this!

Note: the following has not been updated from Leabra version to axon:

Call at the start of the Layer InhibFmGeAct method like this:

```Go
// InhibFmGeAct computes inhibition Gi from Ge and Act averages within relevant Pools
func (ly *Layer) InhibFmGeAct(ltime *Time) {
	lpl := &ly.Pools[0]
	ly.Params.Inhib.Layer.Inhib(&lpl.Inhib)
	ly.InterInhib.Inhib(&ly.Layer) // does inter-layer inhibition
	ly.PoolInhibFmGeAct(ltime)
}
```


