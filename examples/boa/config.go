package main

import (
	"fmt"
	"os"
	"time"

	"github.com/BurntSushi/toml"
)

type Config struct {
	GUI          bool
	PROFILE      bool
	TRAIN_TRIALS int // how many Trials per Epoch
	TRAIN_EPOCHS int // how many Epochs before we exit

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
	config.TRAIN_TRIALS = 100
	config.TRAIN_EPOCHS = 100
}
