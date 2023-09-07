package axon

/*
func TestSetGetUS(t *testing.T) {
	t.Skip("need to update")
	TheNetwork = NewNetwork("TestSetGetUS")
	ctx := NewContext()
	// TODO: seems impossible to test with only one drive
	// because of the +1 for curiosity in PVLVSetUS.
	ctx.PVLV.NPosUSs = 2
	require.NoError(t, TheNetwork.Build(ctx))
	const di = 0
	ctx.PVLVInitUS(di)
	const usIdx = 0
	assert.Equal(t, float32(0), PVLVUSStimVal(ctx, di, usIdx, Positive))
	assert.Equal(t, float32(0), PVLVUSStimVal(ctx, di, usIdx, Negative))
	assert.False(t, PVLVHasPosUS(ctx, di))

	ctx.PVLVSetUS(di, Positive, usIdx, 1)
	// TODO: +1 for curiosity is hard-coded into PVLVSetUS. This assymetry between the
	// getter and setter seems bug-prone.
	const usIdxForGetterPositive = usIdx + 1
	assert.Equal(t, float32(1), PVLVUSStimVal(ctx, 0, usIdxForGetterPositive, Positive))
	assert.True(t, PVLVHasPosUS(ctx, di))

	ctx.PVLVSetUS(di, Negative, usIdx, 1)
	assert.Equal(t, float32(1), PVLVUSStimVal(ctx, di, usIdx, Negative))
}
*/
