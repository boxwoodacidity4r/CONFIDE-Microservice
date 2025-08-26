public class Hello {
    public static void main(String[] args) {
        System.out.println("Hello World");
    }

    public void foo() {
        bar();
    }

    public void bar() {
        System.out.println("bar");
    }
}
