// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package mvmaxon exports axon packages to the mvm Go interpreter
// and provides functions for connecting with Cogent Lab.
package mvmaxon

import (
	"fmt"
	"io"
	"reflect"

	"cogentcore.org/core/core"
	"cogentcore.org/core/mvmcore"
	"cogentcore.org/core/text/textcore"
	"cogentcore.org/lab/goal/interpreter"
	"cogentcore.org/lab/mvmlab"
	"github.com/emer/emergent/v2/mvmemergent"
)

func init() {
	mvmcore.Interpreters["Goal"] = func(out io.Writer) (mvmcore.Interpreter, error) {
		return NewInterpreter(out)
	}
}

// BindTextEditor is [mvmcore.BindTextEditor], with the Goal interpreter
// of this package (with the axon symbols) registered. Content apps whose
// code blocks are compiled into them (by the core tool) set
// content.BindTextEditor to it to run edited code, and list it in the
// Web.DeferredFunctions of their core.toml, so that js builds load mvm
// and its symbols only when code is first edited:
//
//	[Web]
//	  DeferredFunctions = ["github.com/emer/axon/v2/mvmaxon.BindTextEditor"]
//
// This interpreter does not record the variables of Goal math mode in
// tensorfs (unlike Cogent Lab's mvmlab), so the compiled code must not
// either, and the code blocks can use the fsfffb, chans, kinase, and axon
// packages without importing:
//
//	[ContentCode]
//	  GoalRecord = false
//	  Imports = ["github.com/emer/axon/v2/fsfffb", "github.com/emer/axon/v2/chans", "github.com/emer/axon/v2/kinase", "github.com/emer/axon/v2/axon"]
func BindTextEditor(ed *textcore.Editor, parent *core.Frame, language string) {
	mvmcore.BindTextEditor(ed, parent, language)
}

// Interpreter implements [mvmcore.Interpreter] using the [interpreter.Interpreter] for Goal.
type Interpreter struct {
	*interpreter.Interpreter
}

// NewInterpreter returns a new [Interpreter] writing its output to out, with
// the symbols of the axon and emergent packages imported, in addition to those of
// [mvmlab.UseGUI], and the names of [mvmcore.DotPackage], such as b, the
// parent widget of the code.
func NewInterpreter(out io.Writer) (*Interpreter, error) {
	in := &Interpreter{interpreter.NewInterpreter(interpreter.Options{Stdout: out})}
	Load()
	mvmemergent.Load()
	mvmlab.UseGUI(in.Interpreter, Values, mvmemergent.Values, map[string]map[string]reflect.Value{mvmcore.DotPackage: mvmcore.DotValues()})
	in.Config()
	if _, err := in.Interp.Eval("goal", fmt.Sprintf("import . %q\n", mvmcore.DotPackage)); err != nil {
		return nil, err
	}
	return in, nil
}

// Eval runs the given Goal code.
func (in *Interpreter) Eval(src string) (res reflect.Value, err error) {
	res, _, err = in.Interpreter.EvalCode(src)
	return
}
