#!/usr/bin/env goal

// RunInSims runs given function in sims directory
func RunInSims(fun func(d string)) {
	ex := fsx.Dirs("sims")
	cwd := $pwd$
	for _, d := range ex {
		if d[0] == '_' || d == "equations" || d == "kinasesim" {
			continue
		}
		fd := filepath.Join(cwd, "sims", d)
		fmt.Println(d)
		cd {fd}
		fun(d)
	}
}

command build_all {
	go build -v ./...
}

command test_all {
	go test -v ./...
}

command test_long {
	set TEST_LONG true
	RunInSims(func(d string) {
		go test -v -tags multinet
	})
}	

command params_good {
	RunInSims(func(d string) {
		[goal build -v]
		cmd := "./" + d
		[{cmd} -nogui -save-all -good]
	})
}

goalrun.RunCommands(goalrun.Args())

