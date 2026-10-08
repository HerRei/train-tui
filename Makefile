CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra -std=c11
LDFLAGS ?=
PREFIX  ?= /usr/local

BIN := train_tui
SRC := train_tui.c

$(BIN): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Warnings are errors here, so CI catches them.
test: $(SRC)
	$(CC) $(CFLAGS) -Werror -o $(BIN) $(SRC) $(LDFLAGS)
	sh tests/run.sh ./$(BIN)

install: $(BIN)
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 $(BIN) $(DESTDIR)$(PREFIX)/bin/train-tui

uninstall:
	rm -f $(DESTDIR)$(PREFIX)/bin/train-tui

clean:
	rm -f $(BIN)

.PHONY: test install uninstall clean
