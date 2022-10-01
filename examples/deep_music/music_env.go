// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"os"

	"github.com/emer/emergent/env"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/minmax"
	"github.com/goki/ki/ints"
	"gitlab.com/gomidi/midi/v2/smf"
)

// MusicEnv reads in a midi SMF file and presents it as notes
type MusicEnv struct {
	Nm        string          `desc:"name of this environment"`
	Debug     bool            `desc:"emit debugging messages about the music file"`
	Track     int             `desc:"which track to process"`
	MaxSteps  int             `desc:"limit song length to given number of steps, if > 0"`
	UnitsPer  int             `desc:"number of units per localist unit"`
	NoteRange minmax.Int      `desc:"range of notes in given track"`
	NNotes    int             `desc:"number of notes"`
	Song      etable.Table    `desc:"the song encoded into 200 msec increments, with columns as tracks"`
	Time      env.Ctr         `view:"inline" desc:"current time step"`
	Note      etensor.Float32 `desc:"current note"`
}

func (ev *MusicEnv) Name() string { return ev.Nm }
func (ev *MusicEnv) Desc() string { return "" }

func (ev *MusicEnv) TrackInfo(track smf.Track) (name string, ticks int, bpm float64) {
	for _, ev := range track {
		ticks += int(ev.Delta)
		msg := ev.Message
		if msg.Type() == smf.MetaEndOfTrackMsg {
			// ignore
			continue
		}
		switch {
		case msg.GetMetaTrackName(&name): // set the trackname
			// fmt.Printf("track no: %d  name: %s\n", no, name)
		// case msg.GetMetaInstrument(&name): // set the trackname based on instrument name
		// 	fmt.Printf("instr no: %d  name: %s\n", no, name)
		case msg.GetMetaTempo(&bpm):
			// fmt.Printf("bpm: %0.2f\n", bpm)
		}
	}
	if ev.Debug {
		fmt.Printf("track name: %s  ticks %d\n", name, ticks)
	}
	return
}

func (ev *MusicEnv) LoadSong(fname string) error {
	data, err := os.ReadFile(fname)
	if err != nil {
		fmt.Println(err)
		return err
	}
	// read the bytes
	s, err := smf.ReadFrom(bytes.NewReader(data))

	if err != nil {
		fmt.Printf("MIDI error: %s", err.Error())
		return err
	}

	// fmt.Printf("got %v tracks\n", len(s.Tracks))

	sch := etable.Schema{}

	var tslice []int

	var ticks int
	var bpm float64
	for no, track := range s.Tracks {
		name, tick, bp := ev.TrackInfo(track)
		// fmt.Printf("track:\t%d\tlen:\t%d\n", no, len(track))
		if bp > 0 {
			bpm = bp
		}
		if tick == 0 || len(track) < 20 {
			continue
		}
		tslice = append(tslice, no)
		ticks = ints.MaxInt(ticks, tick)
		sch = append(sch, etable.Column{name, etensor.INT64, nil, nil})
	}

	if ev.Debug {
		fmt.Printf("BPM: %g\n", bpm)
	}
	tickPerRow := 120 // 1/16th note
	nrows := ticks / tickPerRow

	if ev.MaxSteps > 0 && nrows > ev.MaxSteps {
		nrows = ev.MaxSteps
	}

	toggleOn := true
	ev.NoteRange.SetInfinity()

	ev.Song.SetFromSchema(sch, nrows)

	for ti, no := range tslice {
		track := s.Tracks[no]
		var tick int
		lastOnRow := -1
		for _, evt := range track {
			tick += int(evt.Delta)
			msg := evt.Message
			if msg.Type() == smf.MetaEndOfTrackMsg {
				// ignore
				continue
			}
			row := tick / tickPerRow
			if row >= nrows {
				break
			}
			var channel, key, vel uint8
			switch {
			case msg.GetNoteOff(&channel, &key, &vel):
				if ev.Debug && row < 20 {
					fmt.Printf("%d\t%d\tnote off:\t%d\n", tick, row, key)
				}
				for ri := lastOnRow + 1; ri <= row; ri++ {
					ev.Song.SetCellFloatIdx(ti, ri, float64(key))
				}
			case msg.GetNoteOn(&channel, &key, &vel):
				if ti == ev.Track {
					ev.NoteRange.FitValInRange(int(key))
				}
				if toggleOn && lastOnRow >= 0 {
					if ev.Debug && row < 20 {
						fmt.Printf("%d\t%d\tnote off:\t%d\n", tick, row, key)
					}
					for ri := lastOnRow + 1; ri <= row; ri++ {
						ev.Song.SetCellFloatIdx(ti, ri, float64(key))
					}
					lastOnRow = -1
				} else {
					lastOnRow = row
					ev.Song.SetCellFloatIdx(ti, row, float64(key))
					if ev.Debug && row < 20 {
						fmt.Printf("%d\t%d\tnote on:\t%d\n", tick, row, key)
					}
				}
			}
		}
	}
	return nil
}

func (ev *MusicEnv) State(element string) etensor.Tensor {
	switch element {
	case "Note":
		return &ev.Note
	}
	return nil
}

// String returns the current state as a string
func (ev *MusicEnv) String() string {
	return ""
}

func (ev *MusicEnv) Config(fname string, maxRows, unitsper int) {
	ev.UnitsPer = unitsper
	ev.MaxSteps = maxRows
	ev.LoadSong(fname)
	ev.NNotes = ev.NoteRange.Range() + 1
	ev.Note.SetShape([]int{ev.UnitsPer, ev.NNotes}, nil, nil)
}

func (ev *MusicEnv) Init(run int) {
	ev.Time.Scale = env.Trial
	ev.Time.Init()
}

func (ev *MusicEnv) Validate() error {
	return nil
}

func (ev *MusicEnv) Step() bool {
	tm := ev.Time.Cur
	if tm > ev.Song.Rows {
		ev.Time.Set(0)
		tm = 0
	}
	key := int(ev.Song.CellFloatIdx(ev.Track, tm))
	ev.Note.SetZeros()
	if key > 0 {
		for ni := 0; ni < ev.UnitsPer; ni++ {
			ev.Note.Set([]int{ni, key - ev.NoteRange.Min}, 1)
		}
	}
	ev.Time.Incr()
	return true
}

func (ev *MusicEnv) Action(element string, input etensor.Tensor) {
	// nop
}

func (ev *MusicEnv) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	switch scale {
	case env.Trial:
		return ev.Time.Query()
	}
	return -1, -1, false
}

// Compile-time check that implements Env interface
var _ env.Env = (*MusicEnv)(nil)
