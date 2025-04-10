// Copyright (c) 2025, Cogent Core. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package yaegiaxon exports axon packages to the yaegi interpreter
// and provides functions for connecting with Cogent Lab.
package yaegiaxon

import (
	"reflect"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/yaegicore"
	"cogentcore.org/lab/goal/interpreter"
	"cogentcore.org/lab/yaegilab/labsymbols"
	"cogentcore.org/lab/yaegilab/tensorsymbols"
	"github.com/cogentcore/yaegi/interp"
	"github.com/emer/emergent/v2/yaegiemergent"
)

func init() {
	yaegicore.Interpreters["Goal"] = func(options interp.Options) yaegicore.Interpreter {
		return NewInterpreter(options)
	}
}

// Interpreter implements [yaegicore.Interpreter] using the [interpreter.Interpreter] for Goal.
type Interpreter struct {
	*interpreter.Interpreter
}

// NewInterpreter returns a new [Interpreter] initialized with the given options.
func NewInterpreter(options interp.Options) *Interpreter {
	return &Interpreter{interpreter.NewInterpreter(options)}
}

func (in *Interpreter) Use(values interp.Exports) error {
	return in.Interp.Use(values)
}

func (in *Interpreter) ImportUsed() {
	errors.Log(in.Use(tensorsymbols.Symbols))
	errors.Log(in.Use(labsymbols.Symbols))
	errors.Log(in.Use(yaegiemergent.Symbols))
	errors.Log(in.Use(Symbols))
	in.Config()
}

func (in *Interpreter) Eval(src string) (res reflect.Value, err error) {
	res, _, err = in.Interpreter.Eval(src)
	return
}
