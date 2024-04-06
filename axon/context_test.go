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
	const usIndex = 0
	assert.Equal(t, float32(0), PVLVUSStimValue(ctx, di, usIndex, Positive))
	assert.Equal(t, float32(0), PVLVUSStimValue(ctx, di, usIndex, Negative))
	assert.False(t, PVLVHasPosUS(ctx, di))

	ctx.PVLVSetUS(di, Positive, usIndex, 1)
	// TODO: +1 for curiosity is hard-coded into PVLVSetUS. This assymetry between the
	// getter and setter seems bug-prone.
	const usIndexForGetterPositive = usIndex + 1
	assert.Equal(t, float32(1), PVLVUSStimValue(ctx, 0, usIndexForGetterPositive, Positive))
	assert.True(t, PVLVHasPosUS(ctx, di))

	ctx.PVLVSetUS(di, Negative, usIndex, 1)
	assert.Equal(t, float32(1), PVLVUSStimValue(ctx, di, usIndex, Negative))
}
*/
