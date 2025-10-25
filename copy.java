// class A{

//     int a = 90;
// }
// class B extends A{
    
//     int a = 34;
//     // this is one will be print but i want the above one to print it also okay
//     void show(){
//         System.out.println(a);
//         System.out.println(super.a);

//     }
// }



// class A{

//     void show(){

//         System.out.println("kamal hai of A");
//     }
// }
// class B extends A{
    
    
//     void show(){
//         super.show();
//         System.out.println("kamal hai of b from a");
       
        
//     }
// }



// we can use the super key word with constructuro aslo 
// class A{
//     A(int a){
//         // super();
//         System.out.println("i am con in A" + a);
//     }

//   }
// class B extends A{
    
//     B(){
//         super(100);
//         System.out.println("i am con from b ");
//     }

    
// }


























// this key word confuse the jvm so we use this for not lettign the jvm to be cofnfused

// class A{
//     void show(){
//         System.out.println(this);
//     }
// }

// class A{
//     int a;
//     A(int a ){
//         this.a = a;
//     }
//     void show(){
//         System.out.println(a);
//     }
// }

class A{
    // int a;
    A( ){
        // this.a = a;
        System.out.println("be with me ");
    }

    A(int a){
        this();
        System.out.println(a);
    }
}
public class copy{

    public static void main(String[] args) {
       
        A r = new A(90);
        // System.out.println(r);
        // r.show();
       
       
        // A obj = new A();
        // System.out.println(obj);
        // obj.show();

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        // B obj= new B();

        // // obj.show();

    }
}