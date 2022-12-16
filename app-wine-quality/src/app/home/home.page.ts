import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-home',
  templateUrl: 'home.page.html',
  styleUrls: ['home.page.scss'],
})
export class HomePage {

  constructor(private http: HttpClient) {}

  resultado = false;
  carregando = false;
  btn_upload = true;
  formData = new FormData();
  resultado_modelo: any = {};


  public async passandoArquivoParaAPI(){

    this.carregando = true;
    this.resultado = true;

    this.resultado_modelo = await this.http.post("http://127.0.0.1:5000/quality-wine/", this.formData).toPromise();
    console.log(this.resultado_modelo);
    
    
    this.carregando = false;
    
      

  }

  
  public carregandoArquivoDoDispositivo(event: any){
    const d = new FormData();
    if (event.target.files && event.target.files[0]){
      const file = event.target.files[0];
      
      d.append('arquivo', file);
      this.formData = d;

      this.btn_upload = false;

    }

  }










}
