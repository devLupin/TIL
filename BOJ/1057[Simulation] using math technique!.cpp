/*
내가 기존에 푼 방법은 큐를 통해 절반씩 뒤로 보내는 방법이다.
연산의 횟수도 너무 많고, 메모리도 많이 차지하여 수학적인 방법으로 해결할 수 있다는 것을 알게 되었다.
*/

#include<iostream>
using namespace std;

int main() {
	int n=0;
	int a, b, c;
	cin >> a >> b >> c;

	while (b != c) {
		b++; c++;
		b /= 2;
		c /= 2;
		n++;
	}
	cout << n;

	return 0;
}