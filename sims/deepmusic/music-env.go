// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deepmusic

import (
	"bytes"
	"fmt"
	"os"
	"time"

	"cogentcore.org/core/math32/minmax"
	"cogentcore.org/lab/table"
	"cogentcore.org/lab/tensor"
	"github.com/emer/emergent/v2/env"
	"gitlab.com/gomidi/midi/v2"
	"gitlab.com/gomidi/midi/v2/gm"
	"gitlab.com/gomidi/midi/v2/smf"
)

// MusicEnv reads in a midi SMF file and presents it as a sequence of notes.
// Songs with one note at a time per track are currently supported.
// Renders note to a tensor with localist note coding with duplicate units for spiking.
type MusicEnv struct {

	// name of this environment
	Name string

	// emit debugging messages about the music file
	Debug bool

	// use only 1 octave of 12 notes for everything -- keeps it consistent
	WrapNotes bool

	// number of time ticks per row in table -- note transitions that are faster than this will be lost
	TicksPer int `default:"120"`

	// which track to process
	Track int

	// play output as it steps
	Play bool

	// limit song length to given number of steps, if > 0
	MaxSteps int

	// time offset for data parallel = Song.Rows / (NData+1)
	DiOffset int `edit:"-"`

	// number of units per localist note value
	UnitsPer int

	// range of notes in given track
	NoteRange minmax.Int

	// number of notes
	NNotes int

	// the song encoded into 200 msec increments, with columns as tracks
	Song *table.Table `display:"-"`

	// current time step
	Time env.Counter `display:"inline"`

	// current note, rendered as a 4D tensor with shape:
	Note tensor.Float32

	// current note index
	NoteIndex int

	// the function for playing midi
	Player func(msg midi.Message) error `display:"-"`

	// for playing notes
	LastNotePlayed int `display:"-"`
}

func (ev *MusicEnv) Label() string { return ev.Name }

func (ev *MusicEnv) Defaults() {
	ev.TicksPer = 120
	ev.WrapNotes = true
}

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
	ev.Song = table.New()
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
		ticks = max(ticks, tick)
		ev.Song.AddIntColumn(name)
	}

	if ev.Debug {
		fmt.Printf("BPM: %g\n", bpm)
	}
	nrows := ticks / ev.TicksPer

	if ev.MaxSteps > 0 && nrows > ev.MaxSteps {
		nrows = ev.MaxSteps
	}

	toggleOn := true
	ev.NoteRange.SetInfinity()

	ev.Song.SetNumRows(nrows)

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
			row := tick / ev.TicksPer
			if row >= nrows {
				break
			}
			var channel, note, vel uint8
			switch {
			case msg.GetNoteOff(&channel, &note, &vel):
				if ev.Debug && row < 20 {
					fmt.Printf("%d\t%d\tnote off:\t%d\n", tick, row, note)
				}
				for ri := lastOnRow + 1; ri <= row; ri++ {
					ev.Song.ColumnByIndex(ti).SetFloat1D(float64(note), ri)
				}
			case msg.GetNoteOn(&channel, &note, &vel):
				if ti == ev.Track {
					ev.NoteRange.FitValInRange(int(note))
				}
				if toggleOn && lastOnRow >= 0 {
					if ev.Debug && row < 20 {
						fmt.Printf("%d\t%d\tnote off:\t%d\n", tick, row, note)
					}
					for ri := lastOnRow + 1; ri <= row; ri++ {
						ev.Song.ColumnByIndex(ti).SetFloat1D(float64(note), ri)
					}
					lastOnRow = -1
				} else {
					lastOnRow = row
					ev.Song.ColumnByIndex(ti).SetFloat1D(float64(note), row)
					if ev.Debug && row < 20 {
						fmt.Printf("%d\t%d\tnote on:\t%d\n", tick, row, note)
					}
				}
			}
		}
	}
	return nil
}

func (ev *MusicEnv) State(element string) tensor.Values {
	switch element {
	case "Note":
		return &ev.Note
	}
	return nil
}

// String returns the current state as a string
func (ev *MusicEnv) String() string {
	return fmt.Sprintf("%d:%d", ev.Time.Cur, ev.NoteIndex)
}

func (ev *MusicEnv) ConfigPlay() error {
	fmt.Printf("outports:\n%s\n", midi.GetOutPorts())

	portname := "IAC Driver Bus 1"

	out, err := midi.FindOutPort(portname)
	if err != nil {
		fmt.Println(err)
		return err
	}
	ev.Player, _ = midi.SendTo(out)
	ev.Player(midi.ProgramChange(0, gm.Instr_Harpsichord.Value()))
	return nil
}

func (ev *MusicEnv) Config(fname string, track, maxRows, unitsper int) {
	if ev.TicksPer == 0 {
		ev.Defaults()
	}
	ev.Track = track
	ev.UnitsPer = unitsper
	ev.MaxSteps = maxRows
	ev.LoadSong(fname)
	ev.NNotes = ev.NoteRange.Range() + 1
	if ev.WrapNotes {
		ev.NNotes = 12
	}
	ev.Note.SetShapeSizes(1, ev.NNotes, ev.UnitsPer, 1)
	if ev.Play {
		ev.ConfigPlay()
	}
}

func (ev *MusicEnv) ConfigNData(ndata int) {
	ev.DiOffset = ev.Song.NumRows() / (ndata + 1)
	if ev.DiOffset < 2 {
		ev.DiOffset = 2
	}
}

func (ev *MusicEnv) Init(run int) {
	ev.Time.Init()
}

func (ev *MusicEnv) Step() bool {
	ev.Time.Incr()
	tm := ev.Time.Cur
	if tm >= ev.Song.NumRows() {
		ev.Time.Set(0)
		tm = 0
	}
	// fmt.Println(ev.Song.NumRows(), ev.Song.NumColumns(), ev.Track)
	// fmt.Println(ev.Song.ColumnByIndex(ev.Track))
	note := int(ev.Song.ColumnByIndex(ev.Track).Float1D(tm))
	ev.RenderNote(note)
	return true
}

// StepDi is data parallel version sampling different offsets from current timestep
func (ev *MusicEnv) StepDi(di int) bool {
	tm := (ev.Time.Cur + di*ev.DiOffset) % ev.Song.NumRows()
	note := int(ev.Song.ColumnByIndex(ev.Track).Float1D(tm))
	ev.RenderNote(note)
	return true
}

func (ev *MusicEnv) RenderNote(note int) {
	ev.NoteIndex = note
	ev.Note.SetZeros()
	if note <= 0 {
		return
	}
	noteidx := note - ev.NoteRange.Min
	// ev.PlayNote(noteidx)
	if ev.WrapNotes {
		noteidx = (note - 9) % 12 // A = 0, etc.
	}
	for ni := 0; ni < ev.UnitsPer; ni++ {
		ev.Note.Set(1, 0, noteidx, ni, 0)
	}
}

// PlayNote actually plays a note (based on index) to the midi device, if Play is active and working
func (ev *MusicEnv) PlayNote(noteIndex int) {
	if !ev.Play || ev.Player == nil {
		return
	}
	note := noteIndex + ev.NoteRange.Min
	if ev.LastNotePlayed > 0 && note != ev.LastNotePlayed {
		ev.Player(midi.NoteOff(0, uint8(ev.LastNotePlayed)))
	}
	if note != ev.LastNotePlayed {
		ev.Player(midi.NoteOn(0, uint8(note), 100))
	}
	time.Sleep(time.Duration(ev.TicksPer) * time.Millisecond)
	ev.LastNotePlayed = note

}

func (ev *MusicEnv) Action(element string, input tensor.Values) {
	// nop
}

// Compile-time check that implements Env interface
var _ env.Env = (*MusicEnv)(nil)
