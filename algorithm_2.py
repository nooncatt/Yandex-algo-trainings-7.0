# Базовые структуры данных: Массивы, списки, стеки, очереди, деки


# Двусвязный список
class MyLinckedList:
    class Node:
        def __init__(self, data):
            self.data = data
            self.prev = self
            self.next = self

        def __str__(self):
            return str(self.data)

        def unlink(self):
            self.next.prev, self.prev.next = self.prev, self.next
            self.prev = self
            self.next = self

    @staticmethod
    def link(nodeleft, noderight):
        nodeleft.next, noderight.prev = noderight, nodeleft

    class NodeIterator:
        def __init__(self, currentNode, sentinel):
            self.currentNode = currentNode
            self.sentinel = sentinel

        def __next__(self):
            if self.currentNode is not None:
                self.currentNode = self.currentNode.next
            if self.currentNode == self.sentinel:
                raise StopIteration

            return self.currentNode

    def __init__(self):
        self.sentinel = MyLinckedList.Node(239)

    def __iter__(self):
        return MyLinckedList.NodeIterator(self.sentinel, self.sentinel)

    def insert(self, position, newNode):
        MyLinckedList.link(newNode, position.next)
        # it should be first otherwise the next of position will be our newNode and it will be looped
        MyLinckedList.link(position, newNode)

    def append(self, newNode):
        self.insert(self.sentinel.prev, newNode)

    def erase(self, currentNode):
        currentNode.unlink()


# simple test
# a = [1, 5, 6, 7]
# m = MyLinckedList()
#
# for x in a:
#     m.append(MyLinckedList.Node(x))
#
# for x in m:
#     print(x)
#
# t = m.sentinel
# t = t.next.next.next.next.next.next
#
# print(t)


# DEQUE, STACK & QUEUE

# QUEUE это очередь: первый кто пришел тот и уйдет первым

# STACK это как корзина с мячами: первый кто пришел уйдет последним


# QUEUE
class Queue:
    class Node:
        def __init__(self, data):
            self.data = data
            self.prev = self
            self.next = self

        def __str__(self):
            return str(self.data)

        def unlink(self):
            self.next.prev, self.prev.next = self.prev, self.next
            self.prev = self
            self.next = self

    @staticmethod
    def link(nodeleft, noderight):
        nodeleft.next, noderight.prev = noderight, nodeleft

    class NodeIterator:
        def __init__(self, currentNode, sentinel):
            self.currentNode = currentNode
            self.sentinel = sentinel

        def __next__(self):
            if self.currentNode is not None:
                self.currentNode = self.currentNode.next
            if self.currentNode == self.sentinel:
                raise StopIteration

            return self.currentNode

    def __init__(self):
        self.sentinel = Queue.Node(239)

    def __iter__(self):
        return Queue.NodeIterator(self.sentinel, self.sentinel)

    def insert(self, position, newNode):
        Queue.link(newNode, self.sentinel)
        # it should be first otherwise the next of position will be our newNode and it will be looped
        Queue.link(position, newNode)

    def append(self, newNode):
        self.insert(self.sentinel.prev, newNode)

    def erase(self):
        self.sentinel.next.unlink()


# STACK
class Stack:
    class Node:
        def __init__(self, data):
            self.data = data
            self.prev = self
            self.next = self

        def __str__(self):
            return str(self.data)

        def unlink(self):
            self.next.prev, self.prev.next = self.prev, self.next
            self.prev = self
            self.next = self

    @staticmethod
    def link(nodeleft, noderight):
        nodeleft.next, noderight.prev = noderight, nodeleft

    class NodeIterator:
        def __init__(self, currentNode, sentinel):
            self.currentNode = currentNode
            self.sentinel = sentinel

        def __next__(self):
            if self.currentNode is not None:
                self.currentNode = self.currentNode.next
            if self.currentNode == self.sentinel:
                raise StopIteration

            return self.currentNode

    def __init__(self):
        self.sentinel = Stack.Node(239)

    def __iter__(self):
        return Stack.NodeIterator(self.sentinel, self.sentinel)

    def insert(self, position, newNode):
        Stack.link(newNode, position.next)
        Stack.link(position, newNode)

    def append(self, newNode):
        self.insert(self.sentinel, newNode)

    def erase(self):
        self.sentinel.next.unlink()


# a = [1, 2, 3, 4]
# m = Stack()
#
# for x in a:
#     m.append(Stack.Node(x))
#
# for x in m:
#     print(x)
#
# m.append(Stack.Node(5))
#
# for x in m:
#     print(x)


from _collections import deque

a = deque([1, 2, 3, 4])
print(a)
a.append(5)
print(a)
a.popleft()
print(a)
