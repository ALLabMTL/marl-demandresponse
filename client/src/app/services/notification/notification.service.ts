import { Injectable } from '@angular/core';
import { MatSnackBar } from '@angular/material/snack-bar';

@Injectable({
  providedIn: 'root',
})
export class NotificationService {
  constructor(private snackBar: MatSnackBar) {}

  openSuccessSnackBar(message: string, action: string) {
    this.snackBar.open(message, action, {
      duration: 3000,
      panelClass: ['green-snackbar', 'login-snackbar'],
      horizontalPosition: 'left',
    });
  }
  openFailureSnackBar(message: string, action: string) {
    this.snackBar.open(message, action, {
      duration: 3000,
      panelClass: ['red-snackbar', 'login-snackbar'],
      horizontalPosition: 'left',
    });
  }
}
