package axon

/*
func TestSetGetUS(t *testing.T) {
	t.Skip("need to update")
	TheNetwork = NewNetwork("TestSetGetUS")
	ctx := NewContext()
	// TODO: seems impossible to test with only one drive
	// because of the +1 for curiosity in .RubiconSetUS.
	ctx.Rubicon.NPosUSs = 2
	require.NoError(t, TheNetwork.Build(ctx))
	const di = 0
	ctx.RubiconInitUS(di)
	const usIndex = 0
	assert.Equal(t, float32(0), .RubiconUSStimValue(ctx, di, usIndex, Positive))
	assert.Equal(t, float32(0), .RubiconUSStimValue(ctx, di, usIndex, Negative))
	assert.False(t, .RubiconHasPosUS(ctx, di))

	ctx.RubiconSetUS(di, Positive, usIndex, 1)
	// TODO: +1 for curiosity is hard-coded into .RubiconSetUS. This assymetry between the
	// getter and setter seems bug-prone.
	const usIndexForGetterPositive = usIndex + 1
	assert.Equal(t, float32(1), .RubiconUSStimValue(ctx, 0, usIndexForGetterPositive, Positive))
	assert.True(t, .RubiconHasPosUS(ctx, di))

	ctx.RubiconSetUS(di, Negative, usIndex, 1)
	assert.Equal(t, float32(1), .RubiconUSStimValue(ctx, di, usIndex, Negative))
}
*/
