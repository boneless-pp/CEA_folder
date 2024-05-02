CC=gcc
CFLAGS=-fPIC -O2
LDFLAGS=-shared
SOURCES=pagerank.c
OBJECTS=$(SOURCES:.c=.o)
LIBRARY=libpagerank.so

all: $(LIBRARY) run_py

$(LIBRARY): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $(OBJECTS)

.c.o:
	$(CC) -c $(CFLAGS) $< -o $@

run_py:
	python3 simulation.py

clean:
	rm -f $(OBJECTS) $(LIBRARY)

.PHONY: all clean run_py
