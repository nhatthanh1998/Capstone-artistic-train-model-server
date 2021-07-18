import event_emitter as events

em = events.EventEmitter()

def raiseStop():
    raise StopIteration

em.on('stop', raiseStop)

i = 0
while True:
    try:
        print("Do")
        if i == 10:
            print("EMIT!!!!!!!!!!!!")
            em.emit("stop")
        i = i + 1
    except StopIteration:
        print("Emit stop")
        break;