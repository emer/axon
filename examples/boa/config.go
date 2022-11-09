package main

import (
	"fmt"
	"os"
	"time"

	"github.com/BurntSushi/toml"
)

type Config struct {
	GUI      bool
	PROFILE  bool
	LIFETIME int32 // how many times the world will call step before we exit

	StartTime time.Time
}

func (config *Config) Load(paths ...string) {
	config.setDefaults()
	for _, path := range paths {
		if len(path) == 0 {
			continue
		}
		_, err := toml.DecodeFile(path, &config)
		if err != nil {
			fmt.Fprintln(os.Stderr, "Config.Load(): ", err)
		}
	}
	config.StartTime = time.Now()
}

func (config *Config) setDefaults() {
	config.GUI = true
	config.PROFILE = false
	config.LIFETIME = 999999999
}
