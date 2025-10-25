// abstract class Animal{
//     void legs(){
//         System.out.println("all animal has fours legs");
//     }

//     abstract void sound();
//     abstract void eat(); 
// }
// class dogs extends Animal{
//     @Override
//     void sound(){
//         System.out.println("mew mew");
//     }
//     void eat(){
//         System.out.println("eating");
//     }
// }

// class cow extends dogs{
//     @Override
//     void sound(){
//         System.out.println("ooooooooo");
//     }
//     void eat(){
//         System.out.println("grsss");
//     }
// }


// interface vehical{
//     String name = "kaaml";

//     void start();
//     void stop();
//     default void color(){
//         System.out.println("lsjksdjfkd");
//     }
//     static  void spedd(){
//         System.out.println("dljfksfs");
//     }
// }


// class Customer implements vehical{
//     @Override
//     public void color(){
//         System.out.println("itis red");
//     }
//     @Override
//     public void stop(){
//         System.out.println("it is stop now");
//     }

//     @Override
//     public void start() {
//         throw new UnsupportedOperationException("Not supported yet.");
//     }
// }

// class student{
//     int rool, marks;
//     String name;
//     void input(){
//         System.out.println("enter you r maerks");
//     }
// }
// class pro extends student{
//     void display(){
//         rool=1; name = "kamal"; marks = 90;
//         System.out.println(rool  + name  + marks);
//     }
// }


// class A{
//     int a,b,c;
//     void add(){
//         a = 10; b= 20; 
//         c = a+b;
//         System.out.println(c);
//     }


//     //  int a,b,c;
//     void sub(){
//         a = 10; b= 20; 
//         c = a-b;
//         System.out.println(c);
//     }
// }

// class B extends A {
//     //  int a,b,c;
//     void mul(){
//         a = 10; b= 20; 
//         c = a*b;
//         System.out.println(c);
//     }    

//     //  int a,b,c;
//     void div(){
//         a = 10; b= 20; 
//         c = a/b;
//         System.out.println(c);
//     }
// }


// class C extends B{
//     //  int a,b,c;
//     void rem(){
//         a = 10; b= 20; 
//         c = a%b;
//         System.out.println(c);
//     }
// }

interface A{
    void show();
}

interface B{
    void show();

}

class large implements A,B{
    public void show(){
        System.out.println("A and B of the classes of A and Class of B ");
    }
}
public class inVSst{
    public static void main(String[] args) {
        large l = new large();
        l.show();

        // C r = new C();
        // r.add();
        // r.sub();
        // r.div();
        // r.rem();
        // r.mul();


        // pro p = new pro();
        // p.input(); p.display();
        // dogs d = new dogs();
        // cow c = new cow();
        // d.sound();
        // d.eat();
        // c.eat();
        // c.sound();
        // Customer c = new Customer();
        // c.start();
        // c.stop();
        // c.color();
        // vehical.spedd();

        
    }
}